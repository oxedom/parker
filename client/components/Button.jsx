const colors = {
  primary: "bg-blue-500 hover:bg-blue-400 text-white",
  secondary: "bg-white hover:bg-gray-200 text-slate-800",
  destructive: "bg-red-500 hover:bg-red-400 text-white",
};

const sizes = {
  sm: "text-sm",
  md: "text-base",
  lg: "text-lg",
};

/**
 * **Example usage:**
 * ```jsx
 * <Button intent="primary" fullWidth>
 *  Click me!
 * </Button>
 * ```
 * @param size - sm, md, lg (determines size of button)
 * @param intent - primary, secondary, destructive (determines color of button)
 * @param fullWidth - true, false (self explanatory)
 */
export default function Button({
  children,
  size = "md",
  intent = "secondary",
  fullWidth = false,
  ...buttonProps
}) {
  const isValidColor = Object.keys(colors).includes(intent);
  const isValidSize = Object.keys(sizes).includes(size);
  if (!isValidColor || !isValidSize)
    throw new Error("Invalid props passed to button component, refer to JSdoc");

  const className = `py-1.5 px-3 font-medium rounded-md transition-colors ${
    colors[intent]
  } ${sizes[size]} ${fullWidth ? "w-full" : ""}`;

  return (
    <button
      {...buttonProps}
      className={className + " " + (buttonProps.className || "")}
    >
      {children}
    </button>
  );
}
