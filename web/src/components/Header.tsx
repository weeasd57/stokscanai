"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useLanguage } from "@/contexts/LanguageContext";
import { Globe, BarChart2, Brain, Activity, Menu, X, User, ChevronDown, ArrowLeftRight, Crown } from "lucide-react";
import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";

export default function Header() {
    const { t } = useLanguage();
    const { user, signOut } = useAuth();
    const pathname = usePathname();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [accountMenuOpen, setAccountMenuOpen] = useState(false);

    useEffect(() => {
        setMobileMenuOpen(false);
        setAccountMenuOpen(false);
    }, [pathname]);

    const navItems = [
        { href: "/", label: t("nav.home"), icon: <BarChart2 className="w-4 h-4" /> },
        { href: "/scanner/ai", label: t("nav.scanner.ai"), icon: <Brain className="w-4 h-4" /> },
        { href: "/scanner/technical", label: t("nav.scanner.tech"), icon: <Activity className="w-4 h-4" /> },
        { href: "/scanner/comparison", label: t("nav.scanner.compare"), icon: <ArrowLeftRight className="w-4 h-4" /> },
        { href: "/pro", label: "Pro", icon: <Crown className="w-4 h-4" /> },
    ];

    return (
        <header className="fixed top-0 left-0 right-0 z-[100] px-6 py-6 md:px-8">
            <div className="mx-auto max-w-[1800px] w-full">
                <div className="flex items-center justify-between rounded-[2rem] border border-white/10 bg-zinc-950/40 backdrop-blur-3xl px-6 py-3.5 shadow-[0_20px_50px_rgba(0,0,0,0.5)] ring-1 ring-white/5 transition-all duration-500 hover:border-white/20">
                    {/* Brand / Logo */}
                    <div className="flex items-center gap-6">
                        <Link href="/" className="group flex items-center gap-3">
                            <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-violet-600 shadow-xl shadow-blue-500/20 transition-all duration-500 group-hover:rotate-12 group-hover:scale-110">
                                <Image
                                    src="/favicon_io/apple-touch-icon.png"
                                    alt="Artoro logo"
                                    width={20}
                                    height={20}
                                    className="rounded-lg"
                                    priority
                                />
                                <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 transition-opacity group-hover:opacity-100" />
                            </div>
                            <div className="flex flex-col min-w-0 sm:flex">
                                <span className="text-base font-bold tracking-tight text-white leading-tight truncate">
                                    {t("app.title")}
                                </span>
                                <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-zinc-500 leading-none hidden xs:block">
                                    {t("header.pro_analysis")}
                                </span>
                            </div>
                        </Link>

                        {/* Desktop Navigation */}
                        <nav className="hidden lg:flex items-center gap-1 ml-4 py-1 px-1 rounded-xl bg-white/5 border border-white/5 flex-wrap max-w-full">
                            {navItems.map((item) => {
                                const isActive = pathname === item.href;
                                return (
                                    <Link
                                        key={item.href}
                                        href={item.href}
                                        className={`relative flex items-center justify-center gap-1.5 rounded-lg px-2 lg:px-3 py-1.5 text-[10px] lg:text-xs font-bold uppercase tracking-wider transition-all duration-300 whitespace-nowrap ${isActive
                                            ? "bg-zinc-100 text-zinc-950 shadow-lg shadow-white/5"
                                            : "text-zinc-500 hover:text-zinc-50 hover:bg-white/5"
                                            }`}
                                    >
                                        {item.icon}
                                        {item.label}
                                        {isActive && (
                                            <span className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-blue-500" />
                                        )}
                                    </Link>
                                );
                            })}
                        </nav>
                    </div>

                    {/* Desktop Actions */}
                    <div className="flex items-center gap-2">
                        {user ? (
                            <div className="relative">
                                <button
                                    onClick={() => setAccountMenuOpen(!accountMenuOpen)}
                                    className={`flex items-center gap-2 h-9 px-3 rounded-xl border transition-all ${accountMenuOpen ? "bg-white/10 border-white/20 text-white" : "bg-white/5 border-white/5 text-zinc-400 hover:text-white"}`}
                                >
                                    <div className="w-5 h-5 rounded-full bg-gradient-to-br from-zinc-700 to-zinc-800 flex items-center justify-center border border-white/10">
                                        <User className="h-3 w-3" />
                                    </div>
                                    <ChevronDown className={`h-3 w-3 transition-transform duration-300 ${accountMenuOpen ? "rotate-180" : ""}`} />
                                </button>

                                {accountMenuOpen && (
                                    <div className="absolute right-0 mt-3 w-56 p-1.5 rounded-2xl border border-white/10 bg-zinc-950/90 backdrop-blur-2xl shadow-2xl animate-in fade-in slide-in-from-top-2 duration-200">
                                        <div className="px-3 py-2 mb-1 border-b border-white/5">
                                            <p className="text-[10px] uppercase tracking-widest font-bold text-zinc-500">{t("account.label")}</p>
                                            <p className="text-xs font-medium text-zinc-300 truncate">{user.email}</p>
                                        </div>
                                        <Link
                                            href="/profile"
                                            className="flex items-center gap-2 px-3 py-2.5 text-sm font-medium text-zinc-400 hover:text-white hover:bg-white/5 rounded-xl transition-all"
                                        >
                                            <User className="h-4 w-4" />
                                            {t("nav.profile")}
                                        </Link>
                                        <button
                                            onClick={() => {
                                                setAccountMenuOpen(false);
                                                void signOut();
                                            }}
                                            className="flex w-full items-center gap-2 px-3 py-2.5 text-sm font-medium text-red-400 hover:bg-red-500/10 rounded-xl transition-all"
                                        >
                                            <X className="h-4 w-4" />
                                            {t("auth.logout")}
                                        </button>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <Link
                                href="/login"
                                className="h-9 px-5 flex items-center rounded-xl bg-white text-zinc-950 text-xs font-bold uppercase tracking-wider hover:bg-zinc-200 transition-all shadow-lg shadow-white/5"
                            >
                                {t("auth.login")}
                            </Link>
                        )}

                        {/* Hamburger */}
                        <button
                            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                            className="lg:hidden h-9 w-9 flex items-center justify-center rounded-xl bg-white/5 border border-white/5 text-white hover:bg-white/10 transition-all"
                        >
                            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
                        </button>
                    </div>
                </div>

                {/* Mobile Menu Panel */}
                {mobileMenuOpen && (
                    <div className="lg:hidden mt-3 rounded-2xl border border-white/5 bg-zinc-950/80 backdrop-blur-2xl p-2 shadow-2xl animate-in slide-in-from-top-4 duration-300 overflow-hidden">
                        <div className="grid grid-cols-1 gap-1">
                            {navItems.map((item) => (
                                <Link
                                    key={item.href}
                                    href={item.href}
                                    className={`flex items-center gap-4 px-4 py-4 rounded-xl text-sm font-bold uppercase tracking-widest transition-all ${pathname === item.href
                                        ? "bg-white text-zinc-950"
                                        : "text-zinc-400 hover:text-white hover:bg-white/5"
                                        }`}
                                >
                                    {item.icon}
                                    {item.label}
                                </Link>
                            ))}

                            <div className="h-px bg-white/5 my-2 mx-4" />

                            <div className="flex items-center justify-end p-2">
                                {!user && (
                                    <Link
                                        href="/login"
                                        className="px-6 py-3 rounded-xl bg-blue-600 text-white text-xs font-bold uppercase tracking-widest"
                                    >
                                        {t("auth.login")}
                                    </Link>
                                )}
                            </div>

                            {user && (
                                <div className="grid grid-cols-2 gap-2 p-2">
                                    <Link
                                        href="/profile"
                                        className="flex items-center justify-center gap-2 h-12 rounded-xl border border-white/10 text-xs font-bold uppercase text-zinc-400"
                                    >
                                        <User className="h-4 w-4" /> {t("nav.profile")}
                                    </Link>
                                    <button
                                        onClick={() => signOut()}
                                        className="flex items-center justify-center gap-2 h-12 rounded-xl bg-red-500/10 text-xs font-bold uppercase text-red-500"
                                    >
                                        {t("auth.logout")}
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </header>
    );
}
