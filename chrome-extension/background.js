chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'startBot') {
    // Native messaging host for Python communication
    const port = chrome.runtime.connectNative('com.solanabot.launcher');
    
    port.onMessage.addListener((msg) => {
      if (msg.status === 'running') {
        sendResponse({ success: true });
      } else {
        sendResponse({ success: false, error: msg.error });
      }
    });

    port.onDisconnect.addListener(() => {
      if (chrome.runtime.lastError) {
        sendResponse({ success: false, error: chrome.runtime.lastError });
      }
    });

    return true; // Keep the message channel open for async response
  }
});