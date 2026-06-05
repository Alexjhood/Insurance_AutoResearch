import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AutoResearch Console",
  description: "Insurance AutoResearch control surface",
};

const NAV = [
  { href: "/", label: "Home" },
  { href: "/runs", label: "Runs" },
  { href: "/leaderboard", label: "Leaderboard" },
  { href: "/launch", label: "Launch" },
  { href: "/monitor", label: "Monitor" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
          <nav className="max-w-7xl mx-auto px-4 h-12 flex items-center gap-6">
            <span className="text-brand font-bold tracking-tight mr-4">AutoResearch</span>
            {NAV.map((n) => (
              <Link
                key={n.href}
                href={n.href}
                className="text-gray-400 hover:text-gray-100 transition-colors text-xs uppercase tracking-wide"
              >
                {n.label}
              </Link>
            ))}
          </nav>
        </header>
        <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
          {children}
        </main>
      </body>
    </html>
  );
}
