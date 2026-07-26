// src/components/shell/Sidebar.jsx — with role-based pages + collapse + logout
import { C } from "../../utils/tokens";
import { LayoutDashboard, Users, ShieldCheck, Wrench, Package, BarChart3, ChevronLeft, LogOut, Activity} from "lucide-react";

var NAV_ITEMS = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "tickets", label: "Tickets", icon: Users },
  { key: "sla", label: "SLA", icon: ShieldCheck },
  { key: "maintenance", label: "Assets", icon: Wrench },
  { key: "inventory", label: "Inventory", icon: Package },
  { key: "executive", label: "Reports", icon: BarChart3 },
];

export default function Sidebar(props) {
  var activePage = props.activePage;
  var onNavigate = props.onNavigate;
  var collapsed = props.collapsed || false;
  var onCollapse = props.onCollapse;
  var allowedPages = props.allowedPages || NAV_ITEMS.map(function(n) { return n.key; });
  var onLogout = props.onLogout;

  return (
    <div style={{
      width: collapsed ? 64 : 220,
      transition: "width 0.2s ease",
      background: C.white,
      borderRight: "1px solid " + C.border,
      display: "flex",
      flexDirection: "column",
      flexShrink: 0,
      overflow: "hidden",
    }}>
      {/* Logo */}
      <div style={{ padding: collapsed ? "16px 12px" : "16px 16px", display: "flex", alignItems: "center", gap: 10, borderBottom: "1px solid " + C.borderLight }}>
        <div style={{
          width: 34, height: 34, borderRadius: 8,
          background: "linear-gradient(135deg, #1E6DD6, #3B82F6)",
          display: "flex", alignItems: "center", justifyContent: "center",
          flexShrink: 0,
        }}>
          <Activity size={18} color="#fff" strokeWidth={2.5} />
        </div>
        {!collapsed && (
          <span style={{ fontWeight: 700, color: C.t9, fontSize: 16, letterSpacing: -0.3 }}>OpsIntel</span>
        )}
      </div>

      {/* Navigation — only show pages the role has access to */}
      <nav style={{ flex: 1, padding: "8px 8px" }}>
        {NAV_ITEMS.filter(function(item) { return allowedPages.indexOf(item.key) >= 0; }).map(function(item) {
          var Icon = item.icon;
          var isActive = activePage === item.key;

          return (
            <button
              key={item.key}
              onClick={function() { onNavigate(item.key); }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                width: "100%",
                padding: collapsed ? "10px 0" : "10px 12px",
                justifyContent: collapsed ? "center" : "flex-start",
                border: "none",
                borderRadius: 8,
                cursor: "pointer",
                fontSize: 14,
                fontWeight: isActive ? 600 : 400,
                color: isActive ? C.brand : C.t6,
                background: isActive ? C.brandLight : "transparent",
                marginBottom: 2,
                borderLeft: isActive ? "3px solid " + C.brand : "3px solid transparent",
                transition: "all 0.15s ease",
              }}
            >
              <Icon size={18} />
              {!collapsed && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Bottom: Collapse + Logout */}
      <div style={{ padding: "8px 8px", borderTop: "1px solid " + C.borderLight }}>
        <button
          onClick={onCollapse}
          style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: collapsed ? "8px 0" : "8px 12px",
            justifyContent: collapsed ? "center" : "flex-start",
            border: "none", background: "transparent",
            cursor: "pointer", fontSize: 13, color: C.t5, width: "100%",
            borderRadius: 6,
          }}
        >
          <ChevronLeft size={15} style={{ transform: collapsed ? "rotate(180deg)" : "none", transition: "transform 0.2s" }} />
          {!collapsed && <span>Collapse</span>}
        </button>
        <button
          onClick={onLogout || function() {}}
          style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: collapsed ? "8px 0" : "8px 12px",
            justifyContent: collapsed ? "center" : "flex-start",
            border: "none", background: "transparent",
            cursor: "pointer", fontSize: 13, color: C.rose, width: "100%",
            borderRadius: 6,
          }}
        >
          <LogOut size={15} />
          {!collapsed && <span>Logout</span>}
        </button>
      </div>
    </div>
  );
}
