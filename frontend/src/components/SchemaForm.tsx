import type { JsonSchema } from '../types'
import styles from '../pages/Pages.module.css'

type Values = Record<string, unknown>

interface Props {
  schema: JsonSchema
  values: Values
  onChange: (next: Values) => void
}

const label = (k: string) => k.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())

/** Renders one input per property of a flat JSON Schema (integer, number, boolean, string, enum). */
export function SchemaForm({ schema, values, onChange }: Props) {
  const props = schema.properties ?? {}
  const required = new Set(schema.required ?? [])
  const set = (k: string, v: unknown) => {
    const next = { ...values }
    if (v === '' || v === undefined) delete next[k]
    else next[k] = v
    onChange(next)
  }

  return (
    <div className={styles.fields}>
      {Object.entries(props).map(([k, p]) => {
        const name = label(k) + (required.has(k) ? ' *' : '')
        if (p.type === 'boolean') {
          return (
            <label key={k} className={styles.check}>
              <input
                type="checkbox"
                checked={values[k] === true}
                onChange={(e) => set(k, e.target.checked)}
              />
              {name}
            </label>
          )
        }
        if (p.enum) {
          return (
            <label key={k} className={styles.field}>
              <span>{name}</span>
              <select value={String(values[k] ?? '')} onChange={(e) => set(k, e.target.value)}>
                <option value="">Choose…</option>
                {p.enum.map((o) => (
                  <option key={String(o)} value={String(o)}>
                    {String(o)}
                  </option>
                ))}
              </select>
            </label>
          )
        }
        if (p.type === 'integer' || p.type === 'number') {
          const isInt = p.type === 'integer'
          return (
            <label key={k} className={styles.field}>
              <span>{name}</span>
              <input
                type="number"
                inputMode={isInt ? 'numeric' : 'decimal'}
                min={p.minimum}
                max={p.maximum}
                step={isInt ? 1 : 'any'}
                value={values[k] === undefined ? '' : String(values[k])}
                onChange={(e) =>
                  set(k, e.target.value === '' ? '' : isInt ? parseInt(e.target.value, 10) : Number(e.target.value))
                }
              />
            </label>
          )
        }
        return (
          <label key={k} className={styles.field}>
            <span>{name}</span>
            <input
              type="text"
              maxLength={p.maxLength}
              value={String(values[k] ?? '')}
              onChange={(e) => set(k, e.target.value)}
            />
          </label>
        )
      })}
    </div>
  )
}
