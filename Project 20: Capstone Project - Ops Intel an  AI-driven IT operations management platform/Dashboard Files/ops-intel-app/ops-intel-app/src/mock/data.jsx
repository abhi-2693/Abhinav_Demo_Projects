// src/mock/data.js
//
// Mock data for the OpsIntel dashboard.
// Every field name matches exactly what the notebooks produce.
// When you connect the real API later, you delete this file
// and fetch from the API instead — the components don't change.


// ─────────────────────────────────────────────
// MODEL 1: NLP Ticket Analytics & Recommendation
// Source notebook: ticket_analytics_recommendation_executed.ipynb
// ─────────────────────────────────────────────

export const tickets_enriched = [
  {
    year_ticket_id: "2025_INC_8842",
    ticket_description: "Primary DC failover alert",
    ticket_created_timestamp: "2025-11-24T09:15:00",
    ticket_status: "In Progress",
    predicted_category: "Security",
    predicted_issue_type: "Failure",
    predicted_priority: "P1",
    classification_confidence: 0.98,
    predicted_resolution_time_minutes: 142.3,
    sla_breach_probability: 0.87,
    recommendation_summary: {
      suggested_resolution: "Failover to secondary DC; escalate to L3 on-call network engineer.",
      recommended_engineer_group: "NetSec",
      expected_resolution_time: 128.4,
    },
    assigned_engineers: [
      { engineer_name: "Sarah Chen", specialization: "NetSec", experience_years: 8 },
      { engineer_name: "Ravi Kumar", specialization: "NetSec", experience_years: 6 },
    ],
    similar_tickets: [
      { year_ticket_id: "2024_INC_7120", similarity_score: 0.89, resolution_time_minutes: 135 },
      { year_ticket_id: "2024_INC_6842", similarity_score: 0.81, resolution_time_minutes: 121 },
      { year_ticket_id: "2023_INC_4119", similarity_score: 0.77, resolution_time_minutes: 118 },
    ],
  },
  {
    year_ticket_id: "2025_INC_8845",
    ticket_description: "Storage array SSD degradation",
    ticket_created_timestamp: "2025-11-24T10:20:00",
    ticket_status: "Open",
    predicted_category: "Hardware",
    predicted_issue_type: "Failure",
    predicted_priority: "P2",
    classification_confidence: 0.94,
    predicted_resolution_time_minutes: 218.1,
    sla_breach_probability: 0.64,
    recommendation_summary: {
      suggested_resolution: "Swap degraded SSD; rebuild RAID volume from secondary.",
      recommended_engineer_group: "Infrastructure",
      expected_resolution_time: 205.2,
    },
    assigned_engineers: [
      { engineer_name: "Marcus Wright", specialization: "Infrastructure", experience_years: 10 },
    ],
    similar_tickets: [
      { year_ticket_id: "2025_INC_8201", similarity_score: 0.84, resolution_time_minutes: 198 },
      { year_ticket_id: "2024_INC_7744", similarity_score: 0.79, resolution_time_minutes: 211 },
    ],
  },
  {
    year_ticket_id: "2025_INC_8849",
    ticket_description: "Slow query on analytics cluster",
    ticket_created_timestamp: "2025-11-24T11:05:00",
    ticket_status: "In Progress",
    predicted_category: "Database",
    predicted_issue_type: "Slow",
    predicted_priority: "P3",
    classification_confidence: 0.87,
    predicted_resolution_time_minutes: 94.8,
    sla_breach_probability: 0.31,
    recommendation_summary: {
      suggested_resolution: "Rebuild index on customer_orders; review query plan.",
      recommended_engineer_group: "DBA",
      expected_resolution_time: 88.1,
    },
    assigned_engineers: [
      { engineer_name: "Elena Rodriguez", specialization: "DBA", experience_years: 7 },
    ],
    similar_tickets: [],
  },
  {
    year_ticket_id: "2025_INC_8851",
    ticket_description: "Internal VPN intermittent drop",
    ticket_created_timestamp: "2025-11-24T11:30:00",
    ticket_status: "Open",
    predicted_category: "Network",
    predicted_issue_type: "Slow",
    predicted_priority: "P2",
    classification_confidence: 0.91,
    predicted_resolution_time_minutes: 167.2,
    sla_breach_probability: 0.58,
    recommendation_summary: {
      suggested_resolution: "Check VPN concentrator logs; validate IKE renegotiation interval.",
      recommended_engineer_group: "NetSec",
      expected_resolution_time: 151.0,
    },
    assigned_engineers: [
      { engineer_name: "David Park", specialization: "NetSec", experience_years: 5 },
    ],
    similar_tickets: [],
  },
  {
    year_ticket_id: "2025_INC_8852",
    ticket_description: "Web Server 02 high CPU sustained",
    ticket_created_timestamp: "2025-11-24T12:00:00",
    ticket_status: "Open",
    predicted_category: "Hardware",
    predicted_issue_type: "Slow",
    predicted_priority: "P3",
    classification_confidence: 0.82,
    predicted_resolution_time_minutes: 74.4,
    sla_breach_probability: 0.22,
    recommendation_summary: {
      suggested_resolution: "Profile worker processes; consider horizontal scale.",
      recommended_engineer_group: "Infrastructure",
      expected_resolution_time: 69.8,
    },
    assigned_engineers: [
      { engineer_name: "Alex Thompson", specialization: "Infrastructure", experience_years: 4 },
    ],
    similar_tickets: [],
  },
  {
    year_ticket_id: "2025_INC_8855",
    ticket_description: "Email relay queue backlog",
    ticket_created_timestamp: "2025-11-24T12:45:00",
    ticket_status: "Open",
    predicted_category: "Software",
    predicted_issue_type: "Slow",
    predicted_priority: "P4",
    classification_confidence: 0.79,
    predicted_resolution_time_minutes: 48.2,
    sla_breach_probability: 0.15,
    recommendation_summary: {
      suggested_resolution: "Restart Postfix service; check disk space on relay server.",
      recommended_engineer_group: "Infrastructure",
      expected_resolution_time: 42.0,
    },
    assigned_engineers: [
      { engineer_name: "Lisa Wang", specialization: "Infrastructure", experience_years: 3 },
    ],
    similar_tickets: [],
  },
  {
    year_ticket_id: "2025_INC_8858",
    ticket_description: "Firewall rule blocking partner API",
    ticket_created_timestamp: "2025-11-24T13:10:00",
    ticket_status: "In Progress",
    predicted_category: "Security",
    predicted_issue_type: "Access",
    predicted_priority: "P2",
    classification_confidence: 0.93,
    predicted_resolution_time_minutes: 56.7,
    sla_breach_probability: 0.41,
    recommendation_summary: {
      suggested_resolution: "Add partner IP range to allowlist; verify with security team.",
      recommended_engineer_group: "NetSec",
      expected_resolution_time: 50.3,
    },
    assigned_engineers: [
      { engineer_name: "Sarah Chen", specialization: "NetSec", experience_years: 8 },
    ],
    similar_tickets: [
      { year_ticket_id: "2024_INC_7890", similarity_score: 0.72, resolution_time_minutes: 45 },
    ],
  },
  {
    year_ticket_id: "2025_INC_8860",
    ticket_description: "Backup job failed on NAS cluster",
    ticket_created_timestamp: "2025-11-24T14:00:00",
    ticket_status: "Open",
    predicted_category: "Hardware",
    predicted_issue_type: "Failure",
    predicted_priority: "P1",
    classification_confidence: 0.96,
    predicted_resolution_time_minutes: 185.0,
    sla_breach_probability: 0.78,
    recommendation_summary: {
      suggested_resolution: "Check NAS controller logs; failover to standby node if primary is unresponsive.",
      recommended_engineer_group: "Infrastructure",
      expected_resolution_time: 170.0,
    },
    assigned_engineers: [
      { engineer_name: "Marcus Wright", specialization: "Infrastructure", experience_years: 10 },
    ],
    similar_tickets: [
      { year_ticket_id: "2024_INC_6200", similarity_score: 0.86, resolution_time_minutes: 190 },
    ],
  },
];


// ─────────────────────────────────────────────
// Ticket volume trend (7-day daily counts)
// Source: base.groupby(date).agg(total, resolved)
// ─────────────────────────────────────────────

export const ticket_volume_trend = [
  { day: "Mon", total: 384, resolved: 352 },
  { day: "Tue", total: 412, resolved: 388 },
  { day: "Wed", total: 398, resolved: 371 },
  { day: "Thu", total: 471, resolved: 441 },
  { day: "Fri", total: 468, resolved: 423 },
  { day: "Sat", total: 198, resolved: 189 },
  { day: "Sun", total: 162, resolved: 151 },
];


// ─────────────────────────────────────────────
// MODEL 2: SLA Breach Predictor
// Source notebook: SLA_Breach_Classification_ISB__1_.ipynb
// ─────────────────────────────────────────────

export const sla_breach_scores = [
  { year_ticket_id: "2025_INC_8842", sla_breach_probability: 0.87, sla_risk_band: "High" },
  { year_ticket_id: "2025_INC_8845", sla_breach_probability: 0.64, sla_risk_band: "High" },
  { year_ticket_id: "2025_INC_8849", sla_breach_probability: 0.31, sla_risk_band: "Watch" },
  { year_ticket_id: "2025_INC_8851", sla_breach_probability: 0.58, sla_risk_band: "High" },
  { year_ticket_id: "2025_INC_8852", sla_breach_probability: 0.22, sla_risk_band: "OnTrack" },
  { year_ticket_id: "2025_INC_8855", sla_breach_probability: 0.15, sla_risk_band: "OnTrack" },
  { year_ticket_id: "2025_INC_8858", sla_breach_probability: 0.41, sla_risk_band: "Watch" },
  { year_ticket_id: "2025_INC_8860", sla_breach_probability: 0.78, sla_risk_band: "High" },
];

export const sla_compliance_trend = [
  { month: "Jan", compliance: 97.2 },
  { month: "Feb", compliance: 97.1 },
  { month: "Mar", compliance: 98.4 },
  { month: "Apr", compliance: 96.8 },
  { month: "May", compliance: 95.1 },
  { month: "Jun", compliance: 97.6 },
];


// ─────────────────────────────────────────────
// System downtime (from MAINTENANCE_HISTORY)
// Grouped by service, last 24h
// ─────────────────────────────────────────────

export const system_downtime = [
  { service: "CRM",     minutes: 142 },
  { service: "ERP",     minutes: 48 },
  { service: "Auth",    minutes: 24 },
  { service: "Email",   minutes: 8 },
  { service: "Storage", minutes: 4 },
];


// ─────────────────────────────────────────────
// Active AI insights (cross-model, composed at API layer)
// Top-N from each model's risk-sorted output
// ─────────────────────────────────────────────

export const active_insights = [
  {
    id: "AID_053",
    severity: "critical",
    title: "Critical Storage Failure",
    body: "Array-04 in US-East-1 reporting redundancy loss. Predicted complete failure in 4 hours.",
    time: "2M AGO",
    source: "Survival model · risk 0.88",
  },
  {
    id: "2025_INC_8842",
    severity: "warning",
    title: "Predictive SLA Breach",
    body: "High latency in Payment Gateway. 7 tickets at risk of breaching response-time SLA.",
    time: "14M AGO",
    source: "SLA breach model · p = 0.87",
  },
  {
    id: "2025_INC_8829",
    severity: "critical",
    title: "API Authentication Timeout",
    body: "Consistent 504 errors on Auth microservice. Affecting 42% of incoming user requests.",
    time: "22M AGO",
    source: "NLP classifier · 98% conf.",
  },
  {
    id: "AID_099",
    severity: "info",
    title: "Database Optimization Complete",
    body: "AI-triggered index rebuild finished on customer_orders. Query latency reduced by 47%.",
    time: "1H AGO",
    source: "Self-healing runbook",
  },
];


// ─────────────────────────────────────────────
// MODEL 3: Asset Failure Survival Model
// Source notebook: Asset_survival_model_pred.ipynb
// ─────────────────────────────────────────────

export const asset_risk_scores = [
  {
    asset_id: "AID_053",
    device_type: "Server",
    model_number: "Dell PowerEdge R750",
    predicted_risk_probability: 0.88,
    predicted_remaining_days_to_failure: 3,
    replacement_needed_by_date: "2026-04-18",
    i_lead_time_days: 7,
  },
  {
    asset_id: "AID_055",
    device_type: "Switch",
    model_number: "Cisco Nexus 9000",
    predicted_risk_probability: 0.65,
    predicted_remaining_days_to_failure: 12,
    replacement_needed_by_date: "2026-04-27",
    i_lead_time_days: 21,
  },
  {
    asset_id: "AID_041",
    device_type: "Server",
    model_number: "HPE ProLiant DL380",
    predicted_risk_probability: 0.42,
    predicted_remaining_days_to_failure: 45,
    replacement_needed_by_date: "2026-05-30",
    i_lead_time_days: 7,
  },
  {
    asset_id: "AID_099",
    device_type: "NAS",
    model_number: "Synology DS3622xs",
    predicted_risk_probability: 0.12,
    predicted_remaining_days_to_failure: 180,
    replacement_needed_by_date: "2026-10-12",
    i_lead_time_days: 14,
  },
  {
    asset_id: "AID_022",
    device_type: "Server",
    model_number: "Lenovo ThinkSystem SR650",
    predicted_risk_probability: 0.05,
    predicted_remaining_days_to_failure: 220,
    replacement_needed_by_date: "2026-11-21",
    i_lead_time_days: 7,
  },
];

export const telemetry_agg = {
  cpu: [
    { time: "00:00", value: 28 },
    { time: "04:00", value: 32 },
    { time: "08:00", value: 58 },
    { time: "12:00", value: 71 },
    { time: "16:00", value: 64 },
    { time: "20:00", value: 42 },
    { time: "23:59", value: 35 },
  ],
  ram: [
    { time: "00:00", value: 52 },
    { time: "04:00", value: 55 },
    { time: "08:00", value: 68 },
    { time: "12:00", value: 74 },
    { time: "16:00", value: 71 },
    { time: "20:00", value: 64 },
    { time: "23:59", value: 58 },
  ],
  thermal: [
    { time: "00:00", value: 48 },
    { time: "04:00", value: 46 },
    { time: "08:00", value: 52 },
    { time: "12:00", value: 61 },
    { time: "16:00", value: 64 },
    { time: "20:00", value: 56 },
    { time: "23:59", value: 50 },
  ],
  kpis: {
    avg_cpu: 42,
    avg_mem: 68,
    avg_temp: 55,
    critical: 3,
  },
};

export const system_alerts = [
  {
    alert_type: "FAILURE RISK",
    asset_id: "AID_053",
    body: "PSU degradation detected. Estimated failure in 72 hours.",
  },
  {
    alert_type: "THERMAL ANOMALY",
    asset_id: "AID_041",
    body: "Core temp exceeding threshold by 12%. Inspect cooling path.",
  },
  {
    alert_type: "STORAGE WEAR",
    asset_id: "AID_099",
    body: "SSD lifespan at 92%. Schedule replacement in Q4.",
  },
];


// ─────────────────────────────────────────────
// MODEL 4: Inventory Procurement Algorithm
// Source notebook: Asset_Inventory_algorithm.ipynb
// ─────────────────────────────────────────────

export const procurement_plan = [
  {
    asset_id: "AID_053",
    device_type: "Server",
    model_number: "Dell PowerEdge R750",
    current_stock_quantity: 12,
    reorder_threshold_quantity: 15,
    safety_stock_quantity: 10,
    lead_time_days: 7,
    unit_cost: 4800,
    asset_needed_within_next_leadtime: 4,
    inventory_deficit: -7,
    to_order_flag: true,
    to_order_quantity: 25,
    after_leadtime_inventory_level: 33,
  },
  {
    asset_id: "AID_083",
    device_type: "Peripheral",
    model_number: "Dell UltraSharp U2723QE",
    current_stock_quantity: 45,
    reorder_threshold_quantity: 10,
    safety_stock_quantity: 8,
    lead_time_days: 3,
    unit_cost: 650,
    asset_needed_within_next_leadtime: 0,
    inventory_deficit: 35,
    to_order_flag: false,
    to_order_quantity: 0,
    after_leadtime_inventory_level: 45,
  },
  {
    asset_id: "AID_055",
    device_type: "Switch",
    model_number: "Cisco Catalyst C9200",
    current_stock_quantity: 2,
    reorder_threshold_quantity: 5,
    safety_stock_quantity: 3,
    lead_time_days: 21,
    unit_cost: 3400,
    asset_needed_within_next_leadtime: 3,
    inventory_deficit: -6,
    to_order_flag: true,
    to_order_quantity: 8,
    after_leadtime_inventory_level: 7,
  },
  {
    asset_id: "AID_091",
    device_type: "Accessory",
    model_number: "Logitech MX Master 3S",
    current_stock_quantity: 8,
    reorder_threshold_quantity: 20,
    safety_stock_quantity: 12,
    lead_time_days: 2,
    unit_cost: 95,
    asset_needed_within_next_leadtime: 2,
    inventory_deficit: -14,
    to_order_flag: true,
    to_order_quantity: 30,
    after_leadtime_inventory_level: 36,
  },
  {
    asset_id: "AID_108",
    device_type: "Component",
    model_number: "Intel Xeon Silver 4210",
    current_stock_quantity: 5,
    reorder_threshold_quantity: 4,
    safety_stock_quantity: 3,
    lead_time_days: 14,
    unit_cost: 420,
    asset_needed_within_next_leadtime: 0,
    inventory_deficit: 1,
    to_order_flag: false,
    to_order_quantity: 0,
    after_leadtime_inventory_level: 5,
  },
];

export const inventory_kpis = {
  total_assets: 1284,
  stock_health_pct: 92.4,
  procurement_cost_mtd: 42850,
  critical_lows: 14,
};

export const category_stock = [
  { category: "Laptops",     current: 48, threshold: 75 },
  { category: "Monitors",    current: 82, threshold: 35 },
  { category: "Network",     current: 18, threshold: 45 },
  { category: "Accessories", current: 40, threshold: 62 },
];


// ─────────────────────────────────────────────
// SLA page specific data
// ─────────────────────────────────────────────

export const breach_by_priority = [
  { priority: "P1", off_hours: 48, business: 38 },
  { priority: "P2", off_hours: 44, business: 36 },
  { priority: "P3", off_hours: 40, business: 32 },
  { priority: "P4", off_hours: 52, business: 44 },
];

export const risk_band_distribution = [
  { band: "High",    count: 38, pct: 29.7 },
  { band: "Watch",   count: 24, pct: 18.8 },
  { band: "OnTrack", count: 66, pct: 51.6 },
];

export const top_risk_features = [
  { feature: "response_utilization_pct", importance: 0.84 },
  { feature: "tel_avg_cpu",              importance: 0.67 },
  { feature: "client_breach_rate",       importance: 0.53 },
  { feature: "is_off_hours",            importance: 0.41 },
  { feature: "warranty_remaining_days",  importance: 0.34 },
];


// ─────────────────────────────────────────────
// CROSS-MODEL: Executive page rollup
// ─────────────────────────────────────────────

export const executive_rollup = {
  savings: "$1.24M",
  downtime_reduction: "94.2%",
  sla_compliance: "99.92%",
  efficiency_score: 88,

  efficiency_trend: [
    { month: "Oct", manual: 82, ai: 84 },
    { month: "Nov", manual: 81, ai: 91 },
    { month: "Dec", manual: 76, ai: 95 },
    { month: "Jan", manual: 74, ai: 96 },
    { month: "Feb", manual: 72, ai: 98 },
    { month: "Mar", manual: 70, ai: 99 },
  ],

  business_health: [
    { label: "Revenue protected",     value: "$4.2M" },
    { label: "Cost of downtime",      value: "$125K" },
    { label: "Resource ROI",          value: "312%" },
    { label: "CapEx optimization",    value: "$840K" },
  ],

  strategic_recommendations: [
    {
      impact: "High impact",
      title: "Infrastructure modernization",
      body: "Upgrade Server Group A-14 to NVMe storage. Predicted to reduce latency by 22% and save $14k/mo in maintenance.",
      source: "Survival model + maintenance history",
    },
    {
      impact: "Medium impact",
      title: "SLA optimization",
      body: "Re-allocate 3 Level-3 engineers to the EU-West region during peak hours (08:00-11:00 UTC) to prevent 15% predicted SLA breaches.",
      source: "SLA breach model + engineer roster",
    },
    {
      impact: "Cost saving",
      title: "License rationalization",
      body: "Auto-decommission 42 unused cloud instances in Development sandbox. Immediate annual saving $118,500.",
      source: "Inventory transactions + utilization telemetry",
    },
  ],

  milestones: [
    { pillar: "Cloud governance AI",      status: "On Track", roi: "+22%", owner: "C. Thompson", pct: 85 },
    { pillar: "Edge compute resilience",  status: "Ahead",    roi: "+14%", owner: "M. Zhao",     pct: 92 },
    { pillar: "SecOps integration",       status: "At Risk",  roi: "N/A",  owner: "D. Sterling",  pct: 45 },
    { pillar: "Data lake optimization",   status: "Done",     roi: "+31%", owner: "S. Patel",     pct: 100 },
    { pillar: "Legacy debt reduction",    status: "On Track", roi: "+8%",  owner: "R. Vance",     pct: 78 },
  ],
};