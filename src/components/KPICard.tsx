import { LucideIcon } from 'lucide-react';

interface KPICardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  variant?: 'default' | 'primary' | 'accent';
  delay?: number;
}

const KPICard = ({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  trend, 
  trendValue,
  variant = 'default',
  delay = 0
}: KPICardProps) => {
  const valueClass = {
    default: 'kpi-value',
    primary: 'kpi-value-primary',
    accent: 'kpi-value-accent'
  }[variant];

  const iconClass = {
    default: 'text-muted-foreground',
    primary: 'text-primary',
    accent: 'text-accent'
  }[variant];

  return (
    <div 
      className="analytics-card opacity-0 animate-fade-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            {title}
          </p>
          <p className={valueClass}>{value}</p>
          {subtitle && (
            <p className="text-sm text-muted-foreground">{subtitle}</p>
          )}
        </div>
        <div className={`p-3 rounded-lg bg-secondary/50 ${iconClass}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
      
      {trend && trendValue && (
        <div className="mt-4 flex items-center gap-2">
          <span className={`text-sm font-medium ${
            trend === 'up' ? 'variance-positive' : 
            trend === 'down' ? 'variance-negative' : 
            'text-muted-foreground'
          }`}>
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'} {trendValue}
          </span>
          <span className="text-xs text-muted-foreground">vs predicted</span>
        </div>
      )}
    </div>
  );
};

export default KPICard;
