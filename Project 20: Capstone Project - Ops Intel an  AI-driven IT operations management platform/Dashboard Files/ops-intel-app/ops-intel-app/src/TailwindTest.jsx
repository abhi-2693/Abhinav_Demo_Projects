import { useState } from "react";
import { Activity, Ticket, Wrench } from "lucide-react";

export default function Test() {
  const [count, setCount] = useState(0);
  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans">
      <h1 className="text-3xl font-bold text-slate-900 mb-4">Tailwind Test</h1>
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <Activity size={18} className="text-blue-600 mb-2" />
          <div className="text-2xl font-semibold text-slate-900">1,284</div>
          <div className="text-sm text-slate-500">Active tickets</div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <Ticket size={18} className="text-emerald-600 mb-2" />
          <div className="text-2xl font-semibold text-slate-900">84%</div>
          <div className="text-sm text-slate-500">Resolution rate</div>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <Wrench size={18} className="text-amber-600 mb-2" />
          <div className="text-2xl font-semibold text-slate-900">3</div>
          <div className="text-sm text-slate-500">Critical alerts</div>
        </div>
      </div>
      <button onClick={() => setCount(c => c + 1)} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
        Clicked {count} times
      </button>
      <p className="mt-4 text-sm text-slate-500">If you see styled cards with a blue button, Tailwind is working.</p>
    </div>
  );
}
