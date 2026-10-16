"""What happens to every request, regardless of which route answers it.

Three concerns that belong together because they all apply unconditionally: a
request gets an id, every response carries the security headers, and an error
is answered in text rather than Flask's HTML. Keeping them out of create_app
leaves that function showing the routes.

The error handler is part of this rather than of the routes: a 404 has no route
by definition, and it still has to obey the content type every other reply
promises.
"""

import uuid

from flask import Flask, Response, g

from src.headers import SECURITY_HEADERS


def install(app: Flask, text_reply) -> None:
    """Attach the request id, the headers and the plain-text error handlers.

    ``text_reply`` is passed in rather than imported to keep this module free
    of a circular import back into src.webapp, which owns it.
    """

    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(500)
    def _plain_error(error):
        """Answer errors in text too.

        Flask's defaults are HTML pages. The front end appends whatever comes
        back to the chat log, so an HTML error document would arrive as a wall
        of markup in a bubble -- and a client told to expect text/plain from
        every other response has no reason to handle HTML from this one.
        """
        status = getattr(error, "code", 500)
        name = getattr(error, "name", "Error")
        return text_reply(f"{name}.", status)

    @app.before_request
    def _assign_request_id() -> None:
        """Tag each request so its log lines can be found from a reply."""
        g.request_id = uuid.uuid4().hex[:8]

    @app.after_request
    def _add_security_headers(response: Response) -> Response:
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        request_id = g.get("request_id")
        if request_id:
            response.headers.setdefault("X-Request-Id", request_id)
        return response
