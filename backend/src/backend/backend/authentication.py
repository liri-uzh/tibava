import logging

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import RemoteUserBackend
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import IntegrityError, transaction


logger = logging.getLogger(__name__)


class EmailRemoteUserBackend(RemoteUserBackend):
    """Authenticate trusted proxy identities and provision local TibavaUser rows."""

    create_unknown_user = True

    def authenticate(self, request, remote_user=None, **kwargs):
        if not remote_user:
            return None

        email = remote_user.strip().lower()
        if not email:
            return None

        try:
            validate_email(email)
        except ValidationError:
            logger.warning("Rejected invalid remote-user email")
            return None

        UserModel = get_user_model()
        username_field = UserModel._meta.get_field(UserModel.USERNAME_FIELD)
        if username_field.max_length and len(email) > username_field.max_length:
            logger.warning("Rejected remote-user email longer than the username field")
            return None

        user = self._get_or_create_user(email)
        if user is None or not self.user_can_authenticate(user):
            return None

        self._update_display_name(request, user)
        return user

    def _get_or_create_user(self, email):
        UserModel = get_user_model()
        matches = list(UserModel._default_manager.filter(email__iexact=email)[:2])
        if len(matches) > 1:
            logger.error("Remote identity matches multiple local email addresses")
            return None
        if matches:
            user = matches[0]
            if user.email != email:
                user.email = email
                user.save(update_fields=["email"])
            return user

        username_matches = list(
            UserModel._default_manager.filter(username__iexact=email)[:2]
        )
        if username_matches:
            user = username_matches[0]
            if user.email and user.email.strip().lower() != email:
                logger.error("Remote identity conflicts with an existing username")
                return None
            if not user.email:
                user.email = email
                user.save(update_fields=["email"])
            return user

        try:
            with transaction.atomic():
                user = UserModel(username=email, email=email)
                user.set_unusable_password()
                user.save()
                return user
        except IntegrityError:
            # A concurrent first request may have created the same identity.
            try:
                return UserModel._default_manager.get(username=email, email=email)
            except UserModel.DoesNotExist:
                logger.exception("Could not provision remote user")
                return None

    @staticmethod
    def _update_display_name(request, user):
        if request is None:
            return

        display_name = request.META.get("HTTP_X_DISPLAY_NAME", "").strip()
        if not display_name:
            return

        parts = display_name.split(maxsplit=1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) == 2 else ""
        changed_fields = []
        if user.first_name != first_name:
            user.first_name = first_name
            changed_fields.append("first_name")
        if user.last_name != last_name:
            user.last_name = last_name
            changed_fields.append("last_name")
        if changed_fields:
            user.save(update_fields=changed_fields)
