"use client";

import { useChat, ChatMessage } from "@/contexts/ChatContext";
import { Send, X, MessageSquare, Sparkles, User, Bot, Loader2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";

export default function ChatWidget() {
    const { isOpen, setIsOpen, messages, sendMessage, isLoading, hasKey } = useChat();
    const [input, setInput] = useState("");
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isOpen]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;
        sendMessage(input);
        setInput("");
    };

    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                className="fixed bottom-6 right-6 h-14 w-14 rounded-full bg-indigo-600 hover:bg-indigo-500 shadow-xl shadow-indigo-600/20 flex items-center justify-center text-white transition-all hover:scale-105 z-50 animate-in fade-in zoom-in duration-300"
            >
                <MessageSquare className="h-7 w-7" />
            </button>
        );
    }

    return (
        <div className="fixed bottom-6 right-6 w-[380px] h-[600px] max-h-[80vh] rounded-2xl bg-zinc-950/95 backdrop-blur-xl border border-zinc-800 shadow-2xl flex flex-col z-50 animate-in fade-in slide-in-from-bottom-5 duration-300 overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900/50">
                <div className="flex items-center gap-2 text-white font-medium">
                    <div className="p-1.5 rounded-lg bg-indigo-500/10">
                        <Sparkles className="h-4 w-4 text-indigo-400" />
                    </div>
                    AI Assistant
                </div>
                <button
                    onClick={() => setIsOpen(false)}
                    className="p-1.5 rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white transition-colors"
                >
                    <X className="h-5 w-5" />
                </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {!hasKey && (
                    <div className="p-3 rounded-xl bg-orange-500/10 border border-orange-500/20 text-xs text-orange-200 mb-4">
                        ⚠️ No API Key configured. Please go to Profile settings to add your Gemini API Key for full functionality.
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div
                        key={idx}
                        className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                    >
                        <div className={`
                            h-8 w-8 rounded-full flex items-center justify-center shrink-0
                            ${msg.role === "user" ? "bg-zinc-800 text-zinc-400" : "bg-indigo-600 text-white"}
                        `}>
                            {msg.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                        </div>
                        <div className={`
                            max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed
                            ${msg.role === "user"
                                ? "bg-zinc-800 text-zinc-100 rounded-tr-none"
                                : "bg-indigo-500/10 border border-indigo-500/20 text-zinc-100 rounded-tl-none"}
                        `}>
                            <div className="whitespace-pre-wrap">{msg.content}</div>
                            {msg.actions && (
                                <div className="mt-3 flex gap-2">
                                    {msg.actions.map((act, i) => (
                                        <button key={i} className="text-xs bg-indigo-500 hover:bg-indigo-600 px-3 py-1.5 rounded-full text-white transition-colors">
                                            {act.label}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex gap-3">
                        <div className="h-8 w-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0">
                            <Bot className="h-4 w-4 text-white" />
                        </div>
                        <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-2xl rounded-tl-none px-4 py-3 flex items-center gap-2">
                            <div className="h-2 w-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                            <div className="h-2 w-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                            <div className="h-2 w-2 bg-indigo-400 rounded-full animate-bounce"></div>
                        </div>
                    </div>
                )}
                <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t border-zinc-800 bg-zinc-900/50">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask anything..."
                        className="flex-1 h-10 rounded-xl bg-zinc-950 border border-zinc-800 px-3 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-indigo-500 focus:outline-none transition-colors"
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className="h-10 w-10 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white flex items-center justify-center transition-all"
                    >
                        {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                    </button>
                </form>
            </div>
        </div>
    );
}
