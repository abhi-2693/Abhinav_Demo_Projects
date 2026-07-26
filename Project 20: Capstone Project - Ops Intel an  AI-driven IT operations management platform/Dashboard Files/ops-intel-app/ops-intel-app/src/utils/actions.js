// src/utils/actions.js
//
// Button action handlers used across all pages.
// Each function either downloads a CSV or shows a placeholder alert.

const API_BASE = "http://localhost:8000";

// ── Export actions (download CSV files) ──

export const exportTickets = () => {
  window.open(`${API_BASE}/api/export/tickets`, "_blank");
};

export const exportAssets = () => {
  window.open(`${API_BASE}/api/export/assets`, "_blank");
};

export const exportInventory = () => {
  window.open(`${API_BASE}/api/export/inventory`, "_blank");
};

// ── Placeholder actions (for buttons that need backend wiring) ──

export const createNewTicket = () => {
  alert("Create New Ticket: This will open a form modal in production.\n\nTo implement: add a POST /api/tickets endpoint to FastAPI and a form component to the Tickets page.");
};

export const addNewAsset = () => {
  alert("New Asset: This will open an asset registration form in production.\n\nTo implement: add a POST /api/assets endpoint to FastAPI.");
};

export const autoScheduleMaintenance = (assetId) => {
  alert(`Auto-Schedule Maintenance for ${assetId || "top-risk asset"}.\n\nTo implement: add a POST /api/maintenance/schedule endpoint that writes to an audit log.`);
};

export const approveReorder = (assetId) => {
  alert(`Approve Reorder for ${assetId}.\n\nTo implement: add a POST /api/procurement/${assetId}/approve endpoint.`);
};
