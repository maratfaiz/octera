// Lightweight inline SVG icons -- no external icon library dependency, matching
// this codebase's preference for staying dependency-free where practical.
// Stroke-based, 24x24 viewBox, inherits color via currentColor so callers can
// style through CSS `color`.

import type { SVGProps } from "react";

export type IconProps = SVGProps<SVGSVGElement>;

function base(children: React.ReactNode, props: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="1em"
      height="1em"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {children}
    </svg>
  );
}

export function WaveLogoIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 30 30" width="1.25em" height="1.25em" fill="none" {...props}>
      <circle cx="15" cy="15" r="13" stroke="currentColor" strokeWidth={2} opacity={0.25} />
      <path
        d="M4 17 C 9 10, 12 22, 15 15 S 21 8, 26 13"
        stroke="currentColor"
        strokeWidth={2}
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function PlusIcon(props: IconProps) {
  return base(<path d="M12 5v14M5 12h14" />, props);
}

export function ClockIcon(props: IconProps) {
  return base(
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3.5 2" />
    </>,
    props,
  );
}

export function UploadIcon(props: IconProps) {
  return base(
    <>
      <path d="M12 16V4M12 4l-4 4M12 4l4 4" />
      <path d="M4 16v2a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-2" />
    </>,
    props,
  );
}

export function ImageIcon(props: IconProps) {
  return base(
    <>
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <circle cx="9" cy="10" r="1.6" />
      <path d="M21 16.5l-5.2-5.2a1.5 1.5 0 0 0-2.1 0L4 20" />
    </>,
    props,
  );
}

export function LayersMapIcon(props: IconProps) {
  return base(
    <>
      <path d="M12 3l9 5-9 5-9-5 9-5Z" />
      <path d="M3 13l9 5 9-5" />
    </>,
    props,
  );
}

export function AlertZoneIcon(props: IconProps) {
  return base(
    <>
      <path d="M12 3.5 21 19.5H3L12 3.5Z" />
      <path d="M12 10v4" />
      <circle cx="12" cy="16.7" r="0.15" fill="currentColor" stroke="currentColor" strokeWidth={1.4} />
    </>,
    props,
  );
}

export function RulerIcon(props: IconProps) {
  return base(
    <>
      <rect x="3" y="8" width="18" height="8" rx="1.5" />
      <path d="M7 8v2.5M11 8v2.5M15 8v2.5M19 8v2.5" />
    </>,
    props,
  );
}

export function ReportIcon(props: IconProps) {
  return base(
    <>
      <path d="M7 3h7l4 4v14a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z" />
      <path d="M14 3v4h4" />
      <path d="M8.5 12h7M8.5 15.5h7M8.5 8.5h3" />
    </>,
    props,
  );
}

export function GaugeIcon(props: IconProps) {
  return base(
    <>
      <path d="M4 15a8 8 0 1 1 16 0" />
      <path d="M12 15l3.5-4.5" />
      <circle cx="12" cy="15" r="0.9" fill="currentColor" />
    </>,
    props,
  );
}

export function ActivityIcon(props: IconProps) {
  return base(<path d="M3 12h4l2.5-7L14 19l2-7h5" />, props);
}

export function LogOutIcon(props: IconProps) {
  return base(
    <>
      <path d="M14 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3" />
      <path d="M10 8l-4 4 4 4M6 12h12" />
    </>,
    props,
  );
}

export function MailIcon(props: IconProps) {
  return base(
    <>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="M3.5 6.5L12 13l8.5-6.5" />
    </>,
    props,
  );
}

export function LockIcon(props: IconProps) {
  return base(
    <>
      <rect x="4.5" y="10.5" width="15" height="10" rx="2" />
      <path d="M8 10.5V7.5a4 4 0 0 1 8 0v3" />
    </>,
    props,
  );
}

export function UserIcon(props: IconProps) {
  return base(
    <>
      <circle cx="12" cy="8" r="3.4" />
      <path d="M4.5 20a7.5 7.5 0 0 1 15 0" />
    </>,
    props,
  );
}

export function ArrowLeftIcon(props: IconProps) {
  return base(<path d="M19 12H5M11 6l-6 6 6 6" />, props);
}
