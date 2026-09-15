from django.contrib.auth.middleware import RemoteUserMiddleware


class ShibbolethRemoteUserMiddleware(RemoteUserMiddleware):
    """Read the trusted X-Remote-User header forwarded by the application nginx."""

    header = "HTTP_X_REMOTE_USER"
