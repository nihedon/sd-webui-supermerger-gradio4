'use strict';

(function() {
    document.addEventListener('DOMContentLoaded', function() {
        (async () => {
            let tab = null;
            while (!tab) {
                tab = gradioApp().getElementById("tab_supermerger");
                if (!tab) {
                    await new Promise((resolve) => setTimeout(resolve, 200));
                }
            }
            return tab;
        })().then(() => {
            gradioApp().getElementById("surpermerger_clear_alert").click();
        });
    });
})();
