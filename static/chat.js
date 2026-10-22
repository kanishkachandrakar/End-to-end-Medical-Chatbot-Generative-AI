// Chat page behaviour. Kept in a file rather than inline so the page can be
// served under a Content-Security-Policy that forbids inline script, and so
// the browser can cache it separately from the HTML.
//
// Values that only the server knows arrive as data- attributes on <body>.
$(document).ready(function () {
    const BOT_AVATAR = document.body.dataset.botAvatar;
    const USER_AVATAR = "https://i.ibb.co/d5b84Xw/Untitled-design.png";
    const $log = $("#messageFormeight");

    function currentTime() {
        const now = new Date();
        const hh = String(now.getHours()).padStart(2, "0");
        const mm = String(now.getMinutes()).padStart(2, "0");
        return hh + ":" + mm;
    }

    // Built with .text() rather than string concatenation so that any
    // angle brackets in the question or in the retrieved context are
    // rendered as characters instead of markup.
    function appendMessage(text, fromUser) {
        const $avatar = $("<div>")
            .addClass("img_cont_msg")
            .append($("<img>")
                .attr("src", fromUser ? USER_AVATAR : BOT_AVATAR)
                .attr("alt", "")
                .addClass("rounded-circle user_img_msg"));

        const $bubble = $("<div>")
            .addClass(fromUser ? "msg_cotainer_send" : "msg_cotainer")
            .text(text)
            .append($("<span>")
                .addClass(fromUser ? "msg_time_send" : "msg_time")
                .text(currentTime()));

        const $row = $("<div>").addClass(
            "d-flex mb-4 " + (fromUser ? "justify-content-end" : "justify-content-start")
        );

        if (fromUser) {
            $row.append($bubble, $avatar);
        } else {
            $row.append($avatar, $bubble);
        }

        $log.append($row);
        $log.scrollTop($log[0].scrollHeight);
        return $row;
    }

    // The server answers every error in text/plain, worded for a reader. A
    // body that is anything else did not come from the server: the Space's
    // router returns an HTML page of its own when the container is starting,
    // restarting or asleep, and a 502 or 504 is the likeliest failure this
    // page will ever see. Pasting that markup into a bubble is the thing the
    // server-side error handler exists to prevent, and the previous version
    // handed it straight to appendMessage.
    function failureText(jqXHR) {
        const type = jqXHR.getResponseHeader("Content-Type") || "";
        const body = String(jqXHR.responseText || "").trim();
        // Every message the server can send here is one sentence from
        // messages.py, so a long body is not one of them whatever it claims.
        if (type.indexOf("text/plain") === 0 && body && body.length <= 300) {
            return body;
        }
        return "Sorry, the server is not reachable right now.";
    }

    // A request takes a few seconds: retrieval, then the LLM. Block a
    // second submit while one is in flight rather than queueing another.
    function setBusy(busy) {
        $("#text").prop("disabled", busy);
        $("#send").prop("disabled", busy);
    }

    // The page loads and looks working even when nothing can be answered:
    // /healthz knows, so ask it once rather than letting the first question
    // come back as an error.
    function showBanner(text) {
        $("#serviceBanner").text(text).prop("hidden", false);
    }

    $.ajax({ type: "GET", url: "/healthz", dataType: "text" })
        .done(function (body) {
            if (String(body).indexOf("index-empty") !== -1) {
                showBanner(
                    "The reference index is empty, so answers are not grounded " +
                    "in the book. Rebuilding it is a server-side fix."
                );
            }
        })
        .fail(function (jqXHR) {
            var body = String(jqXHR.responseText || "");
            if (body.indexOf("chain-unavailable") !== -1) {
                showBanner(
                    "The service cannot reach its search index right now, so " +
                    "questions will not be answered. Please try again later."
                );
            } else if (body.indexOf("index-empty") !== -1) {
                showBanner(
                    "The reference index is empty, so answers are not grounded " +
                    "in the book. Rebuilding it is a server-side fix."
                );
            } else if (jqXHR.status === 0 || jqXHR.status >= 500) {
                // No useful body: either the request never arrived or the
                // server failed in a way /healthz cannot describe. A single
                // blip is not worth a banner, so ask /livez before claiming
                // the service is down -- if the process is answering, the
                // page is fine and the first question will say otherwise.
                $.ajax({ type: "GET", url: "/livez", dataType: "text" }).fail(
                    function () {
                        showBanner(
                            "The service is not responding. Please try again later."
                        );
                    }
                );
            }
        });

    $("#messageArea").on("submit", function (event) {
        event.preventDefault();

        const question = $("#text").val().trim();
        if (!question) {
            return;
        }

        appendMessage(question, true);
        $("#text").val("");
        setBusy(true);

        const $pending = appendMessage("\u2026", false);

        $.ajax({
            type: "POST",
            url: "/get",
            data: { msg: question }
        }).done(function (data) {
            appendMessage(data, false);
        }).fail(function (jqXHR) {
            appendMessage(failureText(jqXHR), false);
        }).always(function () {
            $pending.remove();
            setBusy(false);
            $("#text").trigger("focus");
        });
    });
});
