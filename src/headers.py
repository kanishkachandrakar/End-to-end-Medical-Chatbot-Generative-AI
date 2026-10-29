"""Response headers sent on every reply.

Separate from src/webapp.py so the policy can be read -- and tested against
the page's actual contents -- without the routes in the way. tests/
test_security_headers.py cross-checks every origin here against the template
and chat.js, in both directions.
"""

# Content-Security-Policy. Every origin here is one the page actually loads
# from; a test cross-checks this against the template and chat.js, because a
# policy that silently blocks the stylesheet is worse than none at all.
# Fonts come from use.fontawesome.com's own domain, and the user avatar from
# i.ibb.co. 'self' covers chat.js and style.css.
CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://code.jquery.com https://stackpath.bootstrapcdn.com",
        "style-src 'self' https://stackpath.bootstrapcdn.com https://use.fontawesome.com",
        "font-src 'self' https://use.fontawesome.com",
        # No data: here. Nothing loads one -- not the template, not chat.js,
        # not style.css, and not bootstrap 4.1.3 or Font Awesome 5.5.0, both
        # of whose stylesheets were checked for `url(data:` and have none.
        # It was permitted anyway, and the check that forbids unused sources
        # only looked at https:// ones, so it never saw it.
        "img-src 'self' https://i.ibb.co",
        "connect-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        # Nothing here embeds a plugin or another document, and both are
        # routes an injected tag would otherwise still have.
        "object-src 'none'",
        "frame-src 'none'",
    ]
)

SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    # The replies are text/plain and contain passages from a PDF; without this
    # a browser is free to sniff one as HTML and run it.
    "X-Content-Type-Options": "nosniff",
    # frame-ancestors above covers modern browsers; this covers the rest.
    "X-Frame-Options": "DENY",
    # /get is POST-only, so a question is not in a URL to leak -- but the
    # path a visitor came from still is, and the CDNs have no use for it.
    "Referrer-Policy": "no-referrer",
    # A text box and a button need none of these. Listing them denies the
    # permission rather than leaving it to the browser's default.
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    # Not cross-origin isolation, which this comment used to claim: that is
    # COOP *and* COEP together, and only then does crossOriginIsolated become
    # true. What this one header does is sever the window.opener relationship
    # -- a page that opens this one, or that this one opens, cannot reach into
    # it -- which is worth having on its own and is all that is being had.
    #
    # COEP is deliberately absent. Under require-corp every no-CORS
    # cross-origin subresource needs a Cross-Origin-Resource-Policy header,
    # and the user avatar is an <img> that chat.js builds with no crossorigin
    # attribute, from an origin that sends none:
    #
    #     $ curl -sI https://i.ibb.co/...png
    #     access-control-allow-origin: *      <- CORS, which is not CORP
    #
    # So adding COEP would blank that avatar. It could be made CORS-loaded,
    # since the origin does allow it, but that is a browser-visible change
    # and not one to make from a comment.
    "Cross-Origin-Opener-Policy": "same-origin",
}
