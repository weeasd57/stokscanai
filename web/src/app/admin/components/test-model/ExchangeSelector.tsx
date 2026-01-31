import { Building2, Search, Database, Loader2 } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { DateSymbolResult } from "@/lib/api";

interface ExchangeSelectorProps {
    availableExchanges: string[];
    selectedExchange: string;
    setSelectedExchange: (v: string) => void;
    symbols: DateSymbolResult[];
    symbolsLoading: boolean;
    symbolsError: string | null;
    searchSymbolTerm: string;
    setSearchSymbolTerm: (v: string) => void;
    selectedSymbol: DateSymbolResult | null;
    setSelectedSymbol: (v: DateSymbolResult | null) => void;
}

export default function ExchangeSelector({
    availableExchanges,
    selectedExchange,
    setSelectedExchange,
    symbols,
    symbolsLoading,
    symbolsError,
    searchSymbolTerm,
    setSearchSymbolTerm,
    selectedSymbol,
    setSelectedSymbol,
}: ExchangeSelectorProps) {
    return (
        <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5 space-y-4">
                <div className="flex items-center gap-2 text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500">
                    <Building2 className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" /> Exchange
                </div>
                <Select value={selectedExchange} onValueChange={setSelectedExchange}>
                    <SelectTrigger className="h-10 sm:h-12 rounded-lg sm:rounded-xl border border-white/10 bg-zinc-950/50 px-3 sm:px-4 text-xs font-bold uppercase tracking-widest text-zinc-200 outline-none focus:ring-offset-0 focus:ring-0 transition-colors hover:bg-zinc-950">
                        <SelectValue placeholder="Select exchange" />
                    </SelectTrigger>
                    <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                        {availableExchanges.map((exchangeOpt) => (
                            <SelectItem key={exchangeOpt} value={exchangeOpt} className="text-xs font-bold uppercase tracking-widest">
                                {exchangeOpt}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            </div>

            <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5 space-y-4">
                <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2 sm:gap-3 text-[11px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500">
                        <Database className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" /> Symbols
                    </div>
                    <div className="text-[9px] sm:text-[10px] uppercase tracking-[0.3em] text-zinc-600 bg-zinc-950/50 px-2 sm:px-3 py-1 rounded-full">
                        {symbols.length} symbols
                    </div>
                </div>

                <Label className="flex flex-col gap-2 text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500">
                    Search
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500" />
                        <Input
                            type="text"
                            value={searchSymbolTerm}
                            onChange={(e) => setSearchSymbolTerm(e.target.value)}
                            placeholder="Symbol or name..."
                            className="h-10 sm:h-11 rounded-lg sm:rounded-xl border border-white/10 bg-zinc-950/50 pl-9 sm:pl-10 pr-3 text-xs font-bold text-zinc-200 outline-none transition-colors hover:bg-zinc-950"
                        />
                    </div>
                </Label>

                {symbolsLoading ? (
                    <div className="flex items-center gap-2 text-xs text-zinc-400 py-4">
                        <Loader2 className="h-4 w-4 animate-spin" /> Loading symbols...
                    </div>
                ) : symbolsError ? (
                    <div className="text-xs text-red-400 py-4">{symbolsError}</div>
                ) : symbols.length === 0 ? (
                    <div className="text-xs text-zinc-500 py-4">No symbols found.</div>
                ) : (
                    <Select
                        value={selectedSymbol ? `${selectedSymbol.symbol}|${selectedSymbol.exchange ?? ""}` : ""}
                        onValueChange={(value: string) => {
                            const [symbol, exchangeValue] = value.split("|");
                            const match = symbols.find((s) => s.symbol === symbol && (s.exchange ?? "") === exchangeValue);
                            setSelectedSymbol(match ?? null);
                        }}
                    >
                        <SelectTrigger className="h-10 sm:h-12 rounded-lg sm:rounded-xl border border-white/10 bg-zinc-950/50 px-3 sm:px-4 text-xs font-bold uppercase tracking-widest text-zinc-200 outline-none focus:ring-offset-0 focus:ring-0 transition-colors hover:bg-zinc-950">
                            <SelectValue placeholder="Select symbol" />
                        </SelectTrigger>
                        <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200 max-h-[300px]">
                            {symbols.map((s) => (
                                <SelectItem key={`${s.symbol}-${s.exchange}`} value={`${s.symbol}|${s.exchange ?? ""}`} className="text-xs font-bold uppercase tracking-widest">
                                    <span className="flex items-center justify-between gap-3 w-full">
                                        <span className="truncate">{s.symbol} {s.name ? `- ${s.name}` : ""}</span>
                                        {s.rowCount && <span className="text-zinc-500 text-[10px]">({s.rowCount})</span>}
                                    </span>
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                )}
            </div>
        </div>
    );
}
