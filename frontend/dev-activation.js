// Optional: enable auto Authorization header on all future fetch() calls
DevAuth.wrapGlobalFetch();

document.getElementById('dev-signin').addEventListener('click', () => {
  DevAuth.activate({
    baseUrl: 'http://127.0.0.1:8000',   // your API
    // password: 'demo@local',          // optional; omit to show the prompt
    onGranted: ({ ul }) => {
      // Update your UI as you like
      const who = document.querySelector('#who');
      if (who) who.textContent = `authed${ul ? ' (UL)' : ''}`;
    }
  });
});