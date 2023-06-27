/**
 *
 * @param state - [value, setValue]
 * @param Label - String to display
 * @param setSettingsChange - ??
 */
export default function Slider({
  state,
  label,
  max = 100,
  min = 10,
  step = 1,
  unit = "%",
  setSettingsChange,
}) {
  const [value, setValue] = state;

  return (
    <div className="flex flex-col justify-center text-white">
      <label className="font-bold text-left drop-shadow-sm">{label}</label>
      <div className={`grid grid-cols-[auto_1fr] gap-2 whitespace-nowrap`}>
        <span className="text-left">
          {value}
          {unit}
        </span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          className="w-full mr-4 accent-blue-400"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            setSettingsChange(true);
          }}
        />
      </div>
    </div>
  );
}
