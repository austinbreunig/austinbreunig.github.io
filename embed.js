// Guided-visual embed. One script for every figure.visual. See docs/contracts/cycle2-seams.md (S2).
(function () {
  var TIMEOUT = 4000;

  document.querySelectorAll("figure.visual").forEach(setup);

  function setup(figure) {
    var frame = figure.querySelector(".visual-frame");
    var iframe = frame.querySelector("iframe");
    var fallback = figure.querySelector(".visual-fallback");
    var timer;

    function start() {
      clearTimeout(timer);
      timer = setTimeout(showFallback, TIMEOUT);
    }

    function showFallback() {
      frame.hidden = true;
      fallback.hidden = false;
    }

    function reload() {
      frame.hidden = false;
      fallback.hidden = true;
      // Setting src to itself makes the browser load the iframe again.
      iframe.setAttribute("src", iframe.getAttribute("src"));
      start();
    }

    window.addEventListener("message", function (event) {
      // Sandboxed origin is "null", so trust only the iframe's own window.
      if (event.source !== iframe.contentWindow || !event.data) return;
      if (event.data.type === "visual-ready") {
        clearTimeout(timer);
      } else if (event.data.type === "visual-height" && Number.isInteger(event.data.px)) {
        iframe.style.height = event.data.px + "px";
      }
    });

    figure.querySelector(".visual-reload").addEventListener("click", reload);
    start();
  }
})();
