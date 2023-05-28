import { cva } from "class-variance-authority";

const styles = cva("py-1.5 px-2 font-medium rounded-md transition-colors", {
  variants: {
    intent: {
      primary: "bg-blue-500 hover:bg-blue-400 text-white",
      secondary: "bg-gray-300 text-gray-700",
      tertiary: "bg-white hover:bg-gray-200 text-slate-800",
      destructive: "bg-red-500 hover:bg-red-400 text-white",
    },
    fullWidth: {
      true: "w-full",
    },
  },
});

/**
 * **Example usage:**
 * ```jsx
 * <Button intent="primary" fullWidth>
 *  Click me!
 * </Button>
 * ```
 * @param intent - primary, secondary, tertiary, destructive (determines color of button)
 * @param fullWidth - true, false (self explanatory)
 */
export default function Button({ children, ...buttonProps }) {
  return (
    <button
      {...buttonProps}
      className={styles(buttonProps) + " " + (buttonProps.className || "")}
    >
      {children}
    </button>
  );
}
