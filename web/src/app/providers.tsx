"use client";

import type { ReactNode } from "react";
import { AuthProvider } from "@/contexts/AuthContext";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { WatchlistProvider } from "@/contexts/WatchlistContext";
import { AppStateProvider } from "@/contexts/AppStateContext";
import { ChatProvider } from "@/contexts/ChatContext";

export default function Providers({ children }: { children: ReactNode }) {
  return (
    <AuthProvider>
      <LanguageProvider>
        <WatchlistProvider>
          <AppStateProvider>
            {children}
          </AppStateProvider>
        </WatchlistProvider>
      </LanguageProvider>
    </AuthProvider>
  );
}
