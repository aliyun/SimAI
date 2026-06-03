import { useCallback, useState } from 'react';

interface FileUploadZoneProps {
  readonly accept?: string;
  readonly label?: string;
  readonly onFileContent: (content: string, filename: string) => void;
}

export function FileUploadZone({ accept, label, onFileContent }: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result;
        if (typeof content === 'string') {
          setFileName(file.name);
          onFileContent(content, file.name);
        }
      };
      reader.readAsText(file);
    },
    [onFileContent],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`
        relative border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer
        ${isDragging
          ? 'border-[var(--color-accent-blue)] bg-[var(--color-accent-blue)]/5'
          : 'border-[var(--color-bg-hover)] hover:border-[var(--color-text-muted)]'
        }
      `}
    >
      <input
        type="file"
        accept={accept}
        onChange={handleChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      <div className="text-sm text-[var(--color-text-secondary)]">
        {fileName ? (
          <span className="text-[var(--color-accent-green)]">{fileName}</span>
        ) : (
          label ?? 'Drop file here or click to upload'
        )}
      </div>
    </div>
  );
}
