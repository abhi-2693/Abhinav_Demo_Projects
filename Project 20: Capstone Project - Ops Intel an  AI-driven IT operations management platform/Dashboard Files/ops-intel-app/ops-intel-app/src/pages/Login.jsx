// src/pages/Login.jsx — Full blue, no icon, role cards only
import { useState } from "react";
import { C } from "../utils/tokens";
import { Shield, Users, Eye, ChevronRight, BarChart3, Wrench } from "lucide-react";

var ROLES = {
  admin: { title: "System Administrator", desc: "Full access to all modules and settings", icon: Shield, pages: ["dashboard", "tickets", "sla", "maintenance", "inventory", "executive"], user: { name: "Admin User", dept: "IT Operations", avatar: "AU" }, color: "#1E6DD6" },
  analyst: { title: "Data Analyst", desc: "Dashboards, SLA analytics, and reports", icon: BarChart3, pages: ["dashboard", "sla", "executive"], user: { name: "Priya Sharma", dept: "Analytics Team", avatar: "PS" }, color: "#7C3AED" },
  engineer: { title: "Support Engineer", desc: "Tickets, assets, and maintenance", icon: Wrench, pages: ["dashboard", "tickets", "maintenance"], user: { name: "Rajesh Kumar", dept: "L2 Support", avatar: "RK" }, color: "#059669" },
  manager: { title: "Operations Manager", desc: "SLA, inventory, and executive reports", icon: Users, pages: ["dashboard", "sla", "inventory", "executive"], user: { name: "Neha Gupta", dept: "Service Delivery", avatar: "NG" }, color: "#D97706" },
  viewer: { title: "Executive Viewer", desc: "Read-only dashboard and reports", icon: Eye, pages: ["dashboard", "executive"], user: { name: "VP Operations", dept: "Leadership", avatar: "VP" }, color: "#6B7280" },
};

var PAGE_LABELS = { dashboard: "Dashboard", tickets: "Tickets", sla: "SLA", maintenance: "Assets", inventory: "Inventory", executive: "Reports" };

export default function Login(props) {
  var onLogin = props.onLogin;
  var _h = useState(null); var hoverRole = _h[0]; var setHoverRole = _h[1];

  return (
    <div style={{ minHeight: "100vh", background: C.brand, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>

      <div style={{ textAlign: "center", marginBottom: 40 }}>
        <h1 style={{ fontSize: 40, fontWeight: 800, color: "#fff", margin: "0 0 8px", letterSpacing: -0.5 }}>OpsIntel</h1>
        <p style={{ fontSize: 24, color: "rgba(255,255,255,0.6)", margin: 0 }}>AI-Driven Intelligent IT Operations Platform</p>
      </div>

      <div style={{ fontSize: 22, fontWeight: 600, color: "rgba(255,255,255,0.9)", marginBottom: 20 }}>Select your role to continue</div>

      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", justifyContent: "center", maxWidth: 1100, padding: "0 20px" }}>
        {Object.keys(ROLES).map(function(rk) {
          var role = ROLES[rk];
          var Icon = role.icon;
          var isH = hoverRole === rk;
          return (
            <div key={rk}
              onMouseEnter={function() { setHoverRole(rk); }}
              onMouseLeave={function() { setHoverRole(null); }}
              onClick={function() { onLogin(rk, role); }}
              style={{ width: 190, padding: "24px 18px", borderRadius: 14, cursor: "pointer", background: isH ? "rgba(255,255,255,0.18)" : "rgba(255,255,255,0.08)", border: "1px solid " + (isH ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.12)"), transition: "all 0.2s ease", transform: isH ? "translateY(-4px)" : "none", textAlign: "center" }}>
              <div style={{ width: 44, height: 44, borderRadius: 12, margin: "0 auto 14px", background: "rgba(255,255,255,0.12)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Icon size={20} color="#fff" />
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: "#fff", marginBottom: 4 }}>{role.title}</div>
              <div style={{ fontSize: 16, color: "rgba(255,255,255,0.55)", lineHeight: 1.4, marginBottom: 14, minHeight: 28 }}>{role.desc}</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4, justifyContent: "center", marginBottom: 14 }}>
                {role.pages.map(function(p) { return <span key={p} style={{ fontSize: 12, padding: "2px 6px", borderRadius: 4, background: "rgba(255,255,255,0.12)", color: "rgba(255,255,255,0.8)", fontWeight: 500 }}>{PAGE_LABELS[p]}</span>; })}
              </div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 4, fontSize: 15, fontWeight: 600, color: isH ? "#fff" : "rgba(255,255,255,0.6)" }}>
                {"Enter as " + role.user.name.split(" ")[0]}
                <ChevronRight size={14} />
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: 40, fontSize: 15, color: "rgba(255,255,255,0.25)" }}>ISB AMPBA Capstone Project</div>
    </div>
  );
}
