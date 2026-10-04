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

    // A request takes a few seconds: retrieval, then the LLM. Block a
    // second submit while one is in flight rather than queueing another.
    function setBusy(busy) {
        $("#text").prop("disabled", busy);
        $("#send").prop("disabled", busy);
    }

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
            appendMessage(
                jqXHR.responseText || "Sorry, the server is not reachable right now.",
                false
            );
        }).always(function () {
            $pending.remove();
            setBusy(false);
            $("#text").trigger("focus");
        });
    });
});
