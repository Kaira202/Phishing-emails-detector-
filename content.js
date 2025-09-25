console.log("🚀 Phishing Detector content script loaded (auto mode)");

// send message to background.js
function sendToApi(emailContent) {
    chrome.runtime.sendMessage(
        { action: "checkPhishing", emailContent },
        (response) => {
            if (response && response.success) {
                const data = response.data;
                console.log("Prediction from API:", data);
                if (data.prediction === 1) {
                    showBanner("⚠️ This email looks like PHISHING!", "red");
                } else {
                    showBanner("✅ This email looks safe.", "green");
                }
            } else {
                console.error("Background fetch error:", response?.error);
            }
        }
    );
}

// a small banner injected at top of the email view
function showBanner(message, color) {
    let banner = document.getElementById("phishing-detector-banner");
    if (!banner) {
        banner = document.createElement("div");
        banner.id = "phishing-detector-banner";
        banner.style.position = "fixed";
        banner.style.top = "0";
        banner.style.left = "0";
        banner.style.width = "100%";
        banner.style.zIndex = "9999";
        banner.style.textAlign = "center";
        banner.style.fontSize = "16px";
        banner.style.fontWeight = "bold";
        banner.style.padding = "10px";
        document.body.appendChild(banner);
    }
    banner.textContent = message;
    banner.style.background = color;
    banner.style.color = "white";
}

// safely grab email content
function getEmailContent() {
    const subjectEl = document.querySelector("h2.hP");
    const bodyEl = document.querySelector("div.a3s.aXjCH");
    const subject = subjectEl ? subjectEl.innerText : "";
    const body = bodyEl ? bodyEl.innerText : "";
    return { subject, body };
}

// track last checked content
let lastHash = "";

// function to run on DOM change
function processEmail() {
    const emailContent = getEmailContent();
    if (!emailContent.subject && !emailContent.body) return;

    const hash = (emailContent.subject + "||" + emailContent.body).slice(0, 2000);
    if (hash === lastHash) return; // skip duplicates
    lastHash = hash;

    console.log("📩 Auto-detected email content:", emailContent);
    sendToApi(emailContent);
}

// observer to watch Gmail DOM
const observer = new MutationObserver(() => {
    processEmail();
});

// start observing after small delay
setTimeout(() => {
    observer.observe(document.body, { childList: true, subtree: true });
    console.log("👀 Observing DOM for email content...");
    // initial scan
    processEmail();
}, 2000);
