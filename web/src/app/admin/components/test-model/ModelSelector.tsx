import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { LineChart, Loader2 } from "lucide-react";
import { ModelMetadata } from "./types";

interface ModelSelectorProps {
    models: (string | ModelMetadata)[];
    modelsLoading: boolean;
    modelsError: string | null;
    useMultipleModels: boolean;
    setUseMultipleModels: (v: boolean) => void;
    selectedModel: string | null;
    setSelectedModel: (v: string | null) => void;
    selectedModels: Set<string>;
    setSelectedModels: (v: Set<string>) => void;
}

export default function ModelSelector({
    models,
    modelsLoading,
    modelsError,
    useMultipleModels,
    setUseMultipleModels,
    selectedModel,
    setSelectedModel,
    selectedModels,
    setSelectedModels,
}: ModelSelectorProps) {
    return (
        <div className="rounded-xl sm:rounded-2xl border border-white/5 bg-gradient-to-br from-zinc-800/30 to-zinc-900/30 p-4 sm:p-5">
            <div className="text-[10px] sm:text-xs font-black uppercase tracking-[0.3em] text-zinc-500 flex items-center gap-2 justify-between mb-3">
                <div className="flex items-center gap-2">
                    <LineChart className="h-3 w-3 sm:h-4 sm:w-4 text-indigo-400" />
                    Models
                </div>
                <button
                    onClick={() => {
                        setUseMultipleModels(!useMultipleModels);
                        if (!useMultipleModels) {
                            setSelectedModel(null);
                        } else {
                            setSelectedModels(new Set());
                        }
                    }}
                    className={`px-3 py-1 rounded-lg text-[9px] font-bold transition-all ${useMultipleModels
                        ? "bg-emerald-600/30 text-emerald-400 border border-emerald-500/30"
                        : "bg-zinc-800 text-zinc-400 border border-zinc-700"
                        }`}
                >
                    {useMultipleModels ? "✓ Multi" : "Single"}
                </button>
            </div>
            <div className="mt-3 flex flex-col gap-3">
                {modelsLoading ? (
                    <div className="flex items-center gap-2 text-xs text-zinc-400">
                        <Loader2 className="h-4 w-4 animate-spin" /> Loading...
                    </div>
                ) : modelsError ? (
                    <div className="text-xs text-red-400">{modelsError}</div>
                ) : useMultipleModels ? (
                    <div className="space-y-2 max-h-48 overflow-y-auto custom-scrollbar">
                        {models.map((model) => {
                            const name = typeof model === "string" ? model : model.name;
                            const numFeatures = typeof model === "object" ? model.num_features ?? model.numFeatures : undefined;
                            const trainingSamples = typeof model === "object" ? model.trainingSamples ?? model.training_samples : undefined;
                            return (
                                <label key={name} className="flex items-center gap-2 p-2 rounded-lg hover:bg-zinc-800/50 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={selectedModels.has(name)}
                                        onChange={(e) => {
                                            const newSet = new Set(selectedModels);
                                            if (e.target.checked) newSet.add(name);
                                            else newSet.delete(name);
                                            setSelectedModels(newSet);
                                        }}
                                        className="w-4 h-4 rounded border-zinc-700 bg-zinc-900"
                                    />
                                    <span className="flex-1 flex flex-col gap-0.5 min-w-0">
                                        <span className="text-xs text-zinc-300 truncate font-mono">{name}</span>
                                        {(numFeatures || trainingSamples) && (
                                            <span className="flex flex-wrap gap-1 text-[9px] text-zinc-500">
                                                {numFeatures && (
                                                    <span className="px-1.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-700">
                                                        Feat: {numFeatures}
                                                    </span>
                                                )}
                                                {trainingSamples && (
                                                    <span className="px-1.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-700">
                                                        Samples: {trainingSamples}
                                                    </span>
                                                )}
                                            </span>
                                        )}
                                    </span>
                                </label>
                            );
                        })}
                    </div>
                ) : (
                    <Select value={selectedModel ?? ""} onValueChange={(v: string) => setSelectedModel(v)}>
                        <SelectTrigger className="h-10 sm:h-12 rounded-lg sm:rounded-xl border border-white/10 bg-zinc-950/50 px-3 sm:px-4 text-xs font-bold uppercase tracking-widest text-zinc-200 outline-none focus:ring-offset-0 focus:ring-0 transition-colors hover:bg-zinc-950">
                            <SelectValue placeholder="Select model" />
                        </SelectTrigger>
                        <SelectContent className="bg-zinc-900 border-white/10 text-zinc-200">
                            {models.map((model) => {
                                const name = typeof model === "string" ? model : model.name;
                                const numFeatures = typeof model === "object" ? model.num_features ?? model.numFeatures : undefined;
                                return (
                                    <SelectItem key={name} value={name} className="text-xs font-bold uppercase tracking-widest">
                                        <div className="flex items-center justify-between w-full min-w-[200px]">
                                            <span className="truncate">{name}</span>
                                            {numFeatures && <span className="opacity-50 text-[10px] ml-2">F:{numFeatures}</span>}
                                        </div>
                                    </SelectItem>
                                );
                            })}
                        </SelectContent>
                    </Select>
                )}
            </div>
        </div>
    );
}
