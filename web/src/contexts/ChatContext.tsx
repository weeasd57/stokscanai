"use client";

import { createContext, useContext, useState, ReactNode, useEffect, useCallback } from "react";
import { createSupabaseBrowserClient } from "@/lib/supabase/browser";
import { useAuth } from "@/contexts/AuthContext";
import { useAppState } from "@/contexts/AppStateContext";
import { useRouter } from "next/navigation";

// Simple message type
export type ChatMessage = {
    role: "user" | "assistant" | "system";
    content: string;
    timestamp: number;
    actions?: ChatAction[];
};

type ChatAction = {
    label: string;
    type: "navigate" | "function";
    value: string; // URL or function name
};

interface ChatContextType {
    isOpen: boolean;
    setIsOpen: (v: boolean) => void;
    messages: ChatMessage[];
    sendMessage: (text: string) => Promise<void>;
    isLoading: boolean;
    hasKey: boolean;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export function ChatProvider({ children }: { children: ReactNode }) {
    const { user } = useAuth();
    const { setTechScanner, setAiScanner } = useAppState(); // To control app state via AI
    const router = useRouter();
    const supabase = createSupabaseBrowserClient();

    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);

    // API Keys & Config
    const [openRouterKey, setOpenRouterKey] = useState<string | null>(null);
    const [customRules, setCustomRules] = useState<string | null>(null);
    const [hasKey, setHasKey] = useState(false);

    // Initial Welcome Message
    useEffect(() => {
        setMessages([
            {
                role: "assistant",
                content: "Hello! I am your AI Market Assistant. I can help you analyze stocks, explain indicators, or navigate the app. How can I help today?",
                timestamp: Date.now(),
            }
        ]);
    }, []);

    // Load API Keys
    useEffect(() => {
        if (!user) return;

        async function loadKeys() {
            const { data } = await supabase
                .from("profiles")
                .select("openrouter_api_key, custom_ai_rules")
                .eq("id", user!.id)
                .maybeSingle();

            if (data?.openrouter_api_key) {
                setOpenRouterKey(data.openrouter_api_key);
                setHasKey(true);
            }
            if (data?.custom_ai_rules) {
                setCustomRules(data.custom_ai_rules);
            }
        }
        loadKeys();
    }, [user, supabase]);

    const handleAction = useCallback((action: ChatAction) => {
        if (action.type === "navigate") {
            router.push(action.value);
            setIsOpen(false); // Optional: close on nav
        } else if (action.type === "function") {
            // Handle specific triggers
            if (action.value === "SCAN_TECH_BULLISH") {
                setTechScanner(prev => ({ ...prev, rsiMin: "50", rsiMax: "70", aboveEma200: true }));
                router.push("/scanner/technical");
            }
            // Add more actions here
        }
    }, [router, setTechScanner]);

    const sendMessage = async (text: string) => {
        if (!text.trim()) return;

        const newUserMsg: ChatMessage = { role: "user", content: text, timestamp: Date.now() };
        setMessages(prev => [...prev, newUserMsg]);
        setIsLoading(true);

        try {
            if (!hasKey || !openRouterKey) {
                // Fallback or Mock response if no key
                setTimeout(() => {
                    setMessages(prev => [...prev, {
                        role: "assistant",
                        content: "Please add your OpenRouter API Key in the Profile page to enable real AI responses. For now, I can only provide basic navigation help.",
                        timestamp: Date.now(),
                        actions: [{ label: "Go to Profile", type: "navigate", value: "/profile" }]
                    }]);
                    setIsLoading(false);
                }, 1000);
                return;
            }

            // Real OpenRouter Call (Example implementation, assuming OpenRouter is preferred)
            const systemPrompt = `You are a Stock Market AI Assistant integrated into the "Artoro" app.
            
            APP CONTEXT:
            - Home: Dashboard with popular stocks.
            - AI Scanner: Random Forest predictions.
            - Technical Scanner: Filter by RSI, MACD, etc.
            - Comparison: Compare stocks side-by-side.
            
            USER RULES: ${customRules || "None"}
            
            Be concise, helpful, and professional. Format outputs with Markdown.
            If the user asks to scan for bullish stocks, suggest setting Technical Scanner to RSI > 50 and Price > EMA 200.
            `;

            // Using OpenRouter instead of Gemini
            const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${openRouterKey}`,
                    "HTTP-Referer": window.location.origin,
                    "X-Title": "Artoro"
                },
                body: JSON.stringify({
                    model: "google/gemini-flash-1.5", // Still can use Gemini via OpenRouter if desired, or move to GPT-4o-mini
                    messages: [
                        { role: "system", content: systemPrompt },
                        { role: "user", content: text }
                    ]
                })
            });

            const data = await response.json();
            const replyText = data.choices?.[0]?.message?.content || "Sorry, I couldn't process that.";

            setMessages(prev => [...prev, {
                role: "assistant",
                content: replyText,
                timestamp: Date.now()
            }]);

        } catch (err) {
            setMessages(prev => [...prev, {
                role: "assistant",
                content: "Error communicating with AI service.",
                timestamp: Date.now()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <ChatContext.Provider value={{ isOpen, setIsOpen, messages, sendMessage, isLoading, hasKey }}>
            {children}
        </ChatContext.Provider>
    );
}

export function useChat() {
    const context = useContext(ChatContext);
    if (!context) throw new Error("useChat must be used within ChatProvider");
    return context;
}
