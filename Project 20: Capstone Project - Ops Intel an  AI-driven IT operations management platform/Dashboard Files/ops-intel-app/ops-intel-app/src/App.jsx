// src/App.jsx — with Login page + Role-based access + Back to Login nav
import { useState } from "react";
import { C } from "./utils/tokens";
import Login from "./pages/Login";
import Sidebar from "./components/shell/Sidebar";
import Dashboard from "./pages/Dashboard";
import Tickets from "./pages/Tickets";
import SLA from "./pages/SLA";
import Maintenance from "./pages/Maintenance";
import Inventory from "./pages/Inventory";
import Executive from "./pages/Executive";
import { Search, Bell, Settings, Home } from "lucide-react";

export default function App() {
  var _auth = useState(null); var auth = _auth[0]; var setAuth = _auth[1];
  var _page = useState("dashboard"); var page = _page[0]; var setPage = _page[1];
  var _collapsed = useState(false); var collapsed = _collapsed[0]; var setCollapsed = _collapsed[1];

  if (!auth) {
    return (
      <Login onLogin={function(roleKey, role) {
        setAuth({ role: roleKey, title: role.title, pages: role.pages, user: role.user, color: role.color });
        setPage(role.pages[0] || "dashboard");
      }} />
    );
  }

  var allowedPages = auth.pages || ["dashboard"];

  function handleLogout() {
    setAuth(null);
    setPage("dashboard");
  }

  function handleNavigate(p) {
    if (allowedPages.indexOf(p) >= 0) {
      setPage(p);
    } else {
      alert("Access denied. Your role (" + auth.title + ") does not have access to this module.");
    }
  }

  function renderPage() {
    if (allowedPages.indexOf(page) < 0) {
      return (
        <div style={{ padding: 60, textAlign: "center" }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>🔒</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: C.t9, marginBottom: 8 }}>Access Restricted</div>
          <div style={{ fontSize: 14, color: C.t5, marginBottom: 20 }}>
            {"Your role (" + auth.title + ") does not have access to this page."}
          </div>
        </div>
      );
    }
    switch (page) {
      case "dashboard": return <Dashboard />;
      case "tickets": return <Tickets />;
      case "sla": return <SLA />;
      case "maintenance": return <Maintenance />;
      case "inventory": return <Inventory />;
      case "executive": return <Executive />;
      default: return <Dashboard />;
    }
  }

  return (
    <div style={{ display: "flex", height: "100vh", background: C.bg }}>
      <Sidebar
        activePage={page}
        onNavigate={handleNavigate}
        collapsed={collapsed}
        onCollapse={function() { setCollapsed(!collapsed); }}
        allowedPages={allowedPages}
        onLogout={handleLogout}
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Top bar */}
        <div style={{ height: 56, borderBottom: "1px solid " + C.border, background: C.white, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px", flexShrink: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, background: C.bg, borderRadius: 8, padding: "6px 12px", width: 360 }}>
            <Search size={14} color={C.t4} />
            <input placeholder="Search assets, tickets, or documentation..." style={{ border: "none", background: "transparent", outline: "none", fontSize: 13, color: C.t7, width: "100%" }} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            {/* Back to Login button */}
            <button
              onClick={handleLogout}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "6px 12px", borderRadius: 6, fontSize: 12, fontWeight: 500,
                background: C.bg, color: C.t6, border: "1px solid " + C.border,
                cursor: "pointer",
              }}
            >
              <Home size={13} />
              Switch Role
            </button>

            <div style={{ width: 1, height: 24, background: C.border }} />

            <div style={{ position: "relative", cursor: "pointer" }}>
              <Bell size={18} color={C.t6} />
              <div style={{ position: "absolute", top: -2, right: -2, width: 8, height: 8, borderRadius: "50%", background: C.rose, border: "2px solid " + C.white }} />
            </div>
            <Settings size={18} color={C.t6} style={{ cursor: "pointer" }} />
            <div style={{ width: 1, height: 24, background: C.border }} />
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: C.t9 }}>{auth.user.name}</div>
              <div style={{ fontSize: 11, color: C.t5 }}>{auth.title}</div>
            </div>
            <div style={{
              width: 34, height: 34, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 12, fontWeight: 700, color: "#fff", background: auth.color || C.brand, position: "relative",
            }}>
              {auth.user.avatar}
              <div style={{ position: "absolute", bottom: -1, right: -1, width: 10, height: 10, borderRadius: "50%", background: "#22C55E", border: "2px solid " + C.white }} />
            </div>
          </div>
        </div>

        {/* Page content */}
        <div style={{ flex: 1, overflow: "auto", padding: 24 }}>
          {renderPage()}
          <div style={{ marginTop: 24, paddingTop: 16, borderTop: "1px solid " + C.borderLight, display: "flex", justifyContent: "space-between", fontSize: 12, color: C.t4 }}>
            <span>{"© 2025 OpsIntel — AI-Driven IT Operations. Role: " + auth.title}</span>
            <div style={{ display: "flex", gap: 16 }}>
              <span style={{ cursor: "pointer" }}>Support</span>
              <span style={{ cursor: "pointer" }}>Terms of Service</span>
              <span style={{ cursor: "pointer" }}>Privacy Policy</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
