// components/Navbar.js

import { FiHome, FiImage, FiEdit, FiSearch, FiClock } from "react-icons/fi";
import Link from "next/link";
import { signOut } from "next-auth/react";

export default function Navbar({ activeSection, setActiveSection }) {
  return (
    <div className="bg-blue-500 text-white shadow-lg">
      <div className="container mx-auto flex items-center justify-between py-3 px-6">
        <div className="text-xl font-bold">MonApp</div>
        <nav className="flex space-x-6">
          <NavItem
            label="Images"
            icon={<FiImage />}
            href="/gestion_images"
          />
          <NavItem
            label="Transformations"
            icon={<FiEdit />}
            href="/Transfromations"
          />
          <NavItem
            label="Recherche"
            icon={<FiSearch />}
            href="/recherche"
          />
          <NavItem
            label="Historique"
            icon={<FiClock />}
            active={activeSection === "history"}
            onClick={() => setActiveSection("history")}
          />
          <button
            onClick={() =>  signOut({ callbackUrl: "http://localhost:3000/login" })}
            className="text-white hover:bg-blue-600 px-4 py-2 rounded-md"
          >
            Logout
          </button>
        </nav>
      </div>
    </div>
  );
}

function NavItem({ label, icon, active, onClick, href }) {
  if (href) {
    return (
      <Link
        href={href}
        className={`flex items-center space-x-2 text-white px-4 py-2 rounded-md ${
          active ? "bg-blue-700" : "hover:bg-blue-600"
        }`}
      >
        {icon}
        <span>{label}</span>
      </Link>
    );
  }

  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 text-white px-4 py-2 rounded-md ${
        active ? "bg-blue-700" : "hover:bg-blue-600"
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}
