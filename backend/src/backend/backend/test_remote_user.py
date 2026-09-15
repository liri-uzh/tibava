from django.contrib import auth
from django.test import TestCase, override_settings


REMOTE_USER_MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "backend.middleware.ShibbolethRemoteUserMiddleware",
]


@override_settings(
    MIDDLEWARE=REMOTE_USER_MIDDLEWARE,
    AUTHENTICATION_BACKENDS=["backend.authentication.EmailRemoteUserBackend"],
)
class RemoteUserAuthenticationTests(TestCase):
    def request_user(self, email=None, display_name=None):
        headers = {}
        if email is not None:
            headers["HTTP_X_REMOTE_USER"] = email
        if display_name is not None:
            headers["HTTP_X_DISPLAY_NAME"] = display_name
        return self.client.get("/user/get", **headers)

    def test_first_request_creates_user_with_display_name(self):
        response = self.request_user("user@example.com", "Firstname Lastname")

        self.assertEqual(response.json()["status"], "ok")
        user = auth.get_user_model().objects.get()
        self.assertEqual(user.username, "user@example.com")
        self.assertEqual(user.email, "user@example.com")
        self.assertEqual(user.first_name, "Firstname")
        self.assertEqual(user.last_name, "Lastname")

    def test_subsequent_request_reuses_user_and_updates_display_name(self):
        self.request_user("user@example.com", "Old Name")
        original = auth.get_user_model().objects.get()

        self.request_user("user@example.com", "New Display Name")

        self.assertEqual(auth.get_user_model().objects.count(), 1)
        updated = auth.get_user_model().objects.get()
        self.assertEqual(updated.pk, original.pk)
        self.assertEqual(updated.first_name, "New")
        self.assertEqual(updated.last_name, "Display Name")

    def test_email_is_stripped_and_lowercased(self):
        self.request_user("  User@Example.COM  ")

        user = auth.get_user_model().objects.get()
        self.assertEqual(user.username, "user@example.com")
        self.assertEqual(user.email, "user@example.com")

    def test_existing_user_is_reused_by_email_without_overwriting_limits(self):
        UserModel = auth.get_user_model()
        existing = UserModel.objects.create(
            username="legacy-name", email="User@Example.COM", allowance=123
        )

        self.request_user(" user@example.com ", "Current Name")

        existing.refresh_from_db()
        self.assertEqual(UserModel.objects.count(), 1)
        self.assertEqual(existing.username, "legacy-name")
        self.assertEqual(existing.email, "user@example.com")
        self.assertEqual(existing.allowance, 123)
        self.assertEqual(existing.first_name, "Current")
        self.assertEqual(existing.last_name, "Name")

    def test_missing_remote_user_does_not_create_user(self):
        response = self.request_user(display_name="Untrusted Name")

        self.assertEqual(response.json()["error"]["type"], "not_authenticated")
        self.assertEqual(auth.get_user_model().objects.count(), 0)

    def test_different_emails_create_different_users(self):
        self.request_user("one@example.com")
        self.request_user("two@example.com")

        self.assertEqual(auth.get_user_model().objects.count(), 2)

    def test_provisioned_user_has_no_usable_local_password(self):
        self.request_user("user@example.com")
        user = auth.get_user_model().objects.get()

        self.assertFalse(user.has_usable_password())
        self.assertIsNone(
            auth.authenticate(username="user@example.com", password="anything")
        )
