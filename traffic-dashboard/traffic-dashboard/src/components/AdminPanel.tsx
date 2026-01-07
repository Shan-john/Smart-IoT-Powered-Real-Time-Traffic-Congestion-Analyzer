import React, { useState, useRef } from 'react';
import { useDispatch } from 'react-redux';
import { setView, setTrafficData } from '../store';
import { motion, AnimatePresence } from 'framer-motion';
import { useToast } from '../hooks/use-toast';
import {
    Shield,
    Database,
    Trash2,
    Upload,
    FileJson,
    AlertTriangle,
    CheckCircle,
    X,
    Loader2,
    ArrowLeft,
    Server,
    BarChart3
} from 'lucide-react';
import { deleteFirebaseData, uploadFirebaseData } from '../firebase-config';

// Confirmation Dialog Component
const ConfirmDialog = ({
    isOpen,
    onClose,
    onConfirm,
    title,
    message,
    confirmText = "Delete",
    isLoading = false
}: {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: () => void;
    title: string;
    message: string;
    confirmText?: string;
    isLoading?: boolean;
}) => {
    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
                        onClick={onClose}
                    />
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-3xl p-8 shadow-2xl z-50 w-full max-w-md"
                    >
                        <div className="flex items-center gap-4 mb-6">
                            <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
                                <AlertTriangle className="w-6 h-6 text-red-600" />
                            </div>
                            <div>
                                <h3 className="text-lg font-bold text-slate-900">{title}</h3>
                                <p className="text-sm text-slate-500">{message}</p>
                            </div>
                        </div>
                        <div className="flex gap-3 justify-end">
                            <button
                                onClick={onClose}
                                disabled={isLoading}
                                className="px-5 py-2.5 rounded-xl text-sm font-bold text-slate-600 hover:bg-slate-100 transition-colors disabled:opacity-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={onConfirm}
                                disabled={isLoading}
                                className="px-5 py-2.5 rounded-xl text-sm font-bold text-white bg-red-500 hover:bg-red-600 transition-colors flex items-center gap-2 disabled:opacity-50"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Deleting...
                                    </>
                                ) : (
                                    <>
                                        <Trash2 className="w-4 h-4" />
                                        {confirmText}
                                    </>
                                )}
                            </button>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};

// Data Card Component
const DataCard = ({
    title,
    description,
    icon: Icon,
    path,
    colorScheme,
    onDeleteSuccess
}: {
    title: string;
    description: string;
    icon: React.ElementType;
    path: string;
    colorScheme: { bg: string; border: string; iconBg: string; iconColor: string; badge: string };
    onDeleteSuccess: () => void;
}) => {
    const { toast } = useToast();
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDelete = async () => {
        setIsDeleting(true);
        try {
            await deleteFirebaseData(path);
            toast({
                title: "Data Deleted",
                description: `Successfully deleted all ${title.toLowerCase()}.`,
            });
            onDeleteSuccess();
            setIsDeleteDialogOpen(false);
        } catch (error: any) {
            toast({
                title: "Delete Failed",
                description: error.message || "Failed to delete data.",
            });
        } finally {
            setIsDeleting(false);
        }
    };

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Validate file type
        if (!file.name.endsWith('.json')) {
            toast({
                title: "Invalid File",
                description: "Please select a JSON file.",
            });
            return;
        }

        setIsUploading(true);
        setUploadedFileName(file.name);

        try {
            const text = await file.text();
            const data = JSON.parse(text);

            await uploadFirebaseData(path, data);

            toast({
                title: "Upload Successful",
                description: `Successfully uploaded data to ${title.toLowerCase()}.`,
            });

            // Clear file input
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
            setUploadedFileName(null);
        } catch (error: any) {
            console.error('Upload error:', error);
            toast({
                title: "Upload Failed",
                description: error.message || "Failed to parse or upload JSON file.",
            });
            setUploadedFileName(null);
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className={`bg-white rounded-3xl p-6 shadow-sm border ${colorScheme.border} hover:shadow-lg transition-all duration-300`}
            >
                {/* Header */}
                <div className="flex items-start gap-4 mb-6">
                    <div className={`w-14 h-14 rounded-2xl ${colorScheme.iconBg} flex items-center justify-center`}>
                        <Icon className={`w-7 h-7 ${colorScheme.iconColor}`} />
                    </div>
                    <div className="flex-1">
                        <h3 className="text-lg font-bold text-slate-900">{title}</h3>
                        <span className={`inline-block mt-1 text-xs font-medium ${colorScheme.badge} px-2.5 py-1 rounded-full`}>
                            {description}
                        </span>
                    </div>
                </div>

                {/* Actions */}
                <div className="space-y-4">
                    {/* Delete Section */}
                    <div className="p-4 bg-slate-50 rounded-2xl">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-semibold text-slate-700">Delete All Data</p>
                                <p className="text-xs text-slate-400 mt-0.5">Permanently remove all records</p>
                            </div>
                            <button
                                onClick={() => setIsDeleteDialogOpen(true)}
                                className="flex items-center gap-2 px-4 py-2.5 bg-red-50 hover:bg-red-100 text-red-600 rounded-xl text-sm font-bold transition-colors"
                            >
                                <Trash2 className="w-4 h-4" />
                                Delete All
                            </button>
                        </div>
                    </div>

                    {/* Upload Section */}
                    <div className={`p-4 ${colorScheme.bg} rounded-2xl`}>
                        <div className="flex items-center justify-between mb-3">
                            <div>
                                <p className="text-sm font-semibold text-slate-700">Upload JSON Data</p>
                                <p className="text-xs text-slate-400 mt-0.5">Import data from a JSON file</p>
                            </div>
                        </div>

                        <label
                            className={`relative flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-200 ${isUploading
                                ? 'border-blue-400 bg-blue-50'
                                : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50'
                                }`}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".json"
                                onChange={handleFileSelect}
                                disabled={isUploading}
                                className="hidden"
                            />

                            {isUploading ? (
                                <div className="flex flex-col items-center">
                                    <Loader2 className="w-8 h-8 text-blue-500 animate-spin mb-2" />
                                    <p className="text-sm font-medium text-blue-600">Uploading {uploadedFileName}...</p>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center">
                                    <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
                                        <Upload className="w-5 h-5 text-slate-400" />
                                    </div>
                                    <p className="text-sm font-medium text-slate-600">
                                        <span className="text-blue-500">Click to upload</span> or drag and drop
                                    </p>
                                    <p className="text-xs text-slate-400 mt-1">JSON files only</p>
                                </div>
                            )}
                        </label>
                    </div>
                </div>
            </motion.div>

            <ConfirmDialog
                isOpen={isDeleteDialogOpen}
                onClose={() => setIsDeleteDialogOpen(false)}
                onConfirm={handleDelete}
                title={`Delete ${title}?`}
                message={`This will permanently delete all ${title.toLowerCase()}. This action cannot be undone.`}
                isLoading={isDeleting}
            />
        </>
    );
};

// Main Admin Panel Component
export const AdminPanel = () => {
    const dispatch = useDispatch();

    const handleDeleteSuccess = () => {
        // Reset store data after deletion
        dispatch(setTrafficData({
            vehicleCount: 0,
            time: "N/A",
            congestion: [],
            report: [],
            graph: [],
            daily_summary: [],
            detailed_history: [],
            stats: {
                avg_confidence: 0,
                avg_speed: 0,
                avg_stuck_ratio: 0,
                total_events: 0,
                status_breakdown: []
            }
        }));
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="min-h-screen bg-[#f2f3f5] p-4 md:p-8"
        >
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <button
                        onClick={() => dispatch(setView('dashboard'))}
                        className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-slate-700 mb-4 transition-colors"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        Back to Dashboard
                    </button>

                    <div className="flex items-center gap-4">
                        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg">
                            <Shield className="w-8 h-8 text-white" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Admin Panel</h1>
                            <p className="text-slate-500 mt-1">Manage traffic data and processed data</p>
                        </div>
                    </div>
                </div>

                {/* Warning Banner */}
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="bg-amber-50 border border-amber-200 rounded-2xl p-4 mb-8 flex items-start gap-3"
                >
                    <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="text-sm font-semibold text-amber-800">Caution: Admin Operations</p>
                        <p className="text-xs text-amber-600 mt-0.5">
                            Delete and upload operations affect live data. Make sure to backup important data before making changes.
                        </p>
                    </div>
                </motion.div>

                {/* Data Cards Grid */}
                <div className="grid md:grid-cols-2 gap-6">
                    <DataCard
                        title="Traffic Data"
                        description="Raw sensor data"
                        icon={Server}
                        path="traffic_data"
                        colorScheme={{
                            bg: 'bg-blue-50',
                            border: 'border-blue-100',
                            iconBg: 'bg-gradient-to-br from-blue-500 to-cyan-500',
                            iconColor: 'text-white',
                            badge: 'bg-blue-100 text-blue-700'
                        }}
                        onDeleteSuccess={handleDeleteSuccess}
                    />

                    <DataCard
                        title="Processed Data"
                        description="Analytics & reports"
                        icon={BarChart3}
                        path="processed_data"
                        colorScheme={{
                            bg: 'bg-purple-50',
                            border: 'border-purple-100',
                            iconBg: 'bg-gradient-to-br from-purple-500 to-pink-500',
                            iconColor: 'text-white',
                            badge: 'bg-purple-100 text-purple-700'
                        }}
                        onDeleteSuccess={handleDeleteSuccess}
                    />
                </div>

                {/* Help Section */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="mt-8 bg-white rounded-3xl p-6 shadow-sm border border-slate-100"
                >
                    <h3 className="text-sm font-bold text-slate-900 mb-4 flex items-center gap-2">
                        <FileJson className="w-4 h-4 text-slate-400" />
                        JSON Format Reference
                    </h3>

                    <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-slate-50 rounded-xl p-4">
                            <p className="text-xs font-semibold text-slate-600 mb-2">Traffic Data Format:</p>
                            <pre className="text-xs text-slate-500 overflow-x-auto">
                                {`{
  "DD-MM-YYYY": {
    "-uniqueId": {
      "average_speed": 0,
      "confidence": 0.4,
      "date": "DD-MM-YYYY",
      "reason": "slow moving traffic",
      "status": "Moderate Congestion",
      "stuck_ratio": 0,
      "time": "HH:MM",
      "timestamp": 1767772946.65861,
      "vehicle_count": 0
    }
  }
}`}
                            </pre>
                        </div>

                        <div className="bg-slate-50 rounded-xl p-4">
                            <p className="text-xs font-semibold text-slate-600 mb-2">Processed Data Format:</p>
                            <pre className="text-xs text-slate-500 overflow-x-auto">
                                {`{
  "vehicleCount": 0,
  "time": "1:33",
  "congestion": [
    {"name": "reason", "percentage": 25}
  ],
  "graph": [
    {
      "average_speed": 0,
      "confidence": 0.4,
      "percentage": 25,
      "reason": "description",
      "status": "Moderate Congestion",
      "stuck_ratio": 0,
      "timestamp": "13:32:26,07-01-2026"
    }
  ],
  "report": [
    {"reason": "...", "suggestions": [...]}
  ],
  "daily_summary": [...],
  "detailed_history": [...],
  "stats": {...}
}`}
                            </pre>
                        </div>
                    </div>
                </motion.div>
            </div>
        </motion.div>
    );
};

export default AdminPanel;
