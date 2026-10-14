import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface FormFieldProps {
  label: string;
  name: string;
  type: "input" | "select";
  placeholder?: string;
  options?: { value: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
  inputType?: string;
  delay?: number;
}

export function FormField({
  label,
  name,
  type,
  placeholder,
  options,
  value,
  onChange,
  inputType = "text",
  delay = 0,
}: FormFieldProps) {
  return (
    <div 
      className="space-y-2 animate-fade-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      <Label htmlFor={name} className="text-sm font-medium text-foreground">
        {label}
      </Label>
      {type === "input" ? (
        <Input
          id={name}
          name={name}
          type={inputType}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="h-11 bg-card border-border/60 focus:border-primary focus:ring-primary/20 transition-all"
        />
      ) : (
        <Select value={value} onValueChange={onChange}>
          <SelectTrigger className="h-11 bg-card border-border/60 focus:border-primary focus:ring-primary/20">
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent>
            {options?.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}
