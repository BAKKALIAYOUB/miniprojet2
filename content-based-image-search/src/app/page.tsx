"use client";

import { useState } from "react";
import Navbar from "./components/navBar";

export default function App() {
  const [activeSection, setActiveSection] = useState("dashboard");

  const sections = {
    dashboard: <div>Dashboard Content</div>,
    images: <div>Gestion des Images</div>,
    transformations: <div>Transformations d'Images</div>,
    search: <div>Recherche d'Images</div>,
    history: <div>Historique des Recherches</div>,
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navbar Component */}
      <Navbar activeSection={activeSection} setActiveSection={setActiveSection} />

      {/* Main Content */}
      <div className="flex-1 bg-gray-50 py-6">
        <div className="container mx-auto px-6">{sections[activeSection]}</div>
      </div>
    </div>
  );
}
