import "./globals.css";
import type { ReactNode } from "react";
import Providers from "@/app/providers";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ChatWidget from "@/components/ChatWidget";

export const metadata = {
  title: "elztona",
  description: "Smart AI insights to help you analyze and discover stock opportunities",
  manifest: "/favicon_io/site.webmanifest",
  icons: {
    icon: [
      { url: "/favicon_io/favicon-16x16.png", sizes: "16x16", type: "image/png" },
      { url: "/favicon_io/favicon-32x32.png", sizes: "32x32", type: "image/png" },
    ],
    shortcut: "/favicon_io/favicon.ico",
    apple: "/favicon_io/apple-touch-icon.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-zinc-950 text-zinc-50 antialiased selection:bg-blue-500/30 pt-32">
        <Providers>
          <Header />
          <main className="mx-auto w-full max-w-5xl px-6 pb-12">
            {children}
          </main>
          <Footer />
        </Providers>
      </body>
    </html>
  );
}
