/*alert('If you see this alert, then your custom JavaScript script has run!');
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    toggle_collapse: function(n, is_open) {
        if (n != null) {
          return !is_open;
        }
        return is_open;
    }
});*/
if(!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.clientside = {
    toggle_collapse: function(n, is_open) {
        if (n != null) {
          return !is_open;
        }
        return is_open;
    }
};
