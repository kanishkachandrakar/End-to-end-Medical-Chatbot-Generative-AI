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
        "img-src 'self' https://i.ibb.co data:",
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
    # Questions are in the URL on a GET, so do not leak them to the CDNs.
    "Referrer-Policy": "no-referrer",
    # A text box and a button need none of these. Listing them denies the
    # permission rather than leaving it to the browser's default.
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    # Cross-origin isolation for the page itself; harmless for the CDN assets,
    # which are loaded as subresources rather than documents.
    "Cross-Origin-Opener-Policy": "same-origin",
}
