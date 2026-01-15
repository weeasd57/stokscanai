"use client";

import React, { useState } from "react";

interface StockLogoProps {
    symbol: string;
    logoUrl?: string | null;
    size?: "sm" | "md" | "lg" | "xl";
    className?: string;
}

export default function StockLogo({
    symbol,
    logoUrl,
    size = "md",
    className = ""
}: StockLogoProps) {
    // logoStatus: 'local' -> 'remote' -> 'error'
    const [logoStatus, setLogoStatus] = useState<'local' | 'remote' | 'error'>('local');

    const sizeClasses = {
        sm: "h-6 w-6 text-[10px]",
        md: "h-8 w-8 text-xs",
        lg: "h-10 w-10 text-sm",
        xl: "h-14 w-14 text-lg",
    };

    const currentSize = sizeClasses[size];
    const initial = symbol.charAt(0).toUpperCase();

    // Generate a consistent color based on symbol
    const colors = [
        "bg-indigo-600", "bg-emerald-600", "bg-amber-600",
        "bg-rose-600", "bg-cyan-600", "bg-purple-600",
        "bg-blue-600", "bg-orange-600"
    ];
    const colorIndex = symbol.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
    const bgColor = colors[colorIndex];

    // Source selection logic
    let src = `/logos/${symbol}.svg`;
    if (logoStatus === 'remote') src = logoUrl || "";

    const handleError = () => {
        if (logoStatus === 'local') {
            if (logoUrl) setLogoStatus('remote');
            else setLogoStatus('error');
        } else {
            setLogoStatus('error');
        }
    };

    if (logoStatus === 'error' || (!logoUrl && logoStatus === 'remote')) {
        return (
            <div className={`flex-shrink-0 flex items-center justify-center rounded-xl font-black text-white ${bgColor} ${currentSize} ${className} shadow-inner`}>
                {initial}
            </div>
        );
    }

    return (
        <div className={`flex-shrink-0 overflow-hidden rounded-xl bg-white flex items-center justify-center ${currentSize} ${className} border border-white/10 shadow-sm`}>
            <img
                src={src}
                alt={symbol}
                className="w-full h-full object-contain p-1"
                onError={handleError}
            />
        </div>
    );
}
