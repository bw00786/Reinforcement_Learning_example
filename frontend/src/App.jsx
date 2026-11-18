import { useEffect, useMemo, useState } from "react";

const initialConfig = {
  episodes: 10,
  seed: 42,
  agents: 3,
  learning_rate: 0.0003,
  grid_size: 8,
  max_steps: 100,
};

async function readRun(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

function MetricCard({ label, value, note, accent }) {
  return (
    <article className={`metric-card ${accent ?? ""}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </article>
  );
}

function RewardChart({ episodes }) {
  const points = useMemo(() => {
    if (!episodes.length) return "";
    const values = episodes.map((item) => item.cooperative_reward);
    const min = Math.min(0, ...values);
    const max = Math.max(1, ...values);
    const spread = max - min || 1;
    return values
      .map((value, index) => {
        const x = values.length === 1 ? 350 : 30 + (index / (values.length - 1)) * 640;
        const y = 180 - ((value - min) / spread) * 150;
        return `${x},${y}`;
      })
      .join(" ");
  }, [episodes]);

  return (
    <div className="chart-wrap">
      {episodes.length ? (
        <svg className="reward-chart" viewBox="0 0 700 210" role="img" aria-label="Cooperative reward by episode">
          <line x1="30" y1="180" x2="670" y2="180" className="chart-axis" />
          <line x1="30" y1="30" x2="30" y2="180" className="chart-axis" />
          <line x1="30" y1="105" x2="670" y2="105" className="chart-grid" />
          <polyline points={points} className="chart-line" />
          {episodes.map((item, index) => {
            const values = episodes.map((episode) => episode.cooperative_reward);
            const min = Math.min(0, ...values);
            const max = Math.max(1, ...values);
            const y = 180 - ((item.cooperative_reward - min) / (max - min || 1)) * 150;
            const x = values.length === 1 ? 350 : 30 + (index / (values.length - 1)) * 640;
            return <circle key={item.episode} cx={x} cy={y} r="3.5" className="chart-dot" />;
          })}
          <text x="30" y="202" className="chart-label">EP 01</text>
          <text x="670" y="202" textAnchor="end" className="chart-label">
            EP {String(episodes.at(-1).episode).padStart(2, "0")}
          </text>
        </svg>
      ) : (
        <div className="empty-chart">
          <span className="empty-chart-icon">⌁</span>
          <span>Your reward curve will appear here</span>
          <small>Start a training run to begin collecting episode data.</small>
        </div>
      )}
    </div>
  );
}

function App() {
  const [config, setConfig] = useState(initialConfig);
  const [run, setRun] = useState(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    fetch("/api/runs/current")
      .then(readRun)
      .then((current) => {
        if (active && current) setRun(current);
      })
      .catch((cause) => {
        if (active) setError(`Could not connect to the trainer API: ${cause.message}`);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!run?.id || run.status !== "running") return undefined;
    let active = true;
    let timer;
    const poll = async () => {
      try {
        const updated = await readRun(await fetch(`/api/runs/${run.id}`));
        if (active) {
          setRun(updated);
          if (updated.status === "running") timer = window.setTimeout(poll, 900);
        }
      } catch (cause) {
        if (active) {
          setError(`Unable to refresh training progress: ${cause.message}`);
          timer = window.setTimeout(poll, 2000);
        }
      }
    };
    timer = window.setTimeout(poll, 900);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [run?.id, run?.status]);

  const episodes = run?.episodes ?? [];
  const averageReward = episodes.length
    ? episodes.reduce((sum, item) => sum + item.cooperative_reward, 0) / episodes.length
    : 0;
  const harvested = episodes.reduce((sum, item) => sum + item.harvested, 0);
  const latest = episodes.at(-1);
  const status = run?.status ?? "ready";

  function updateConfig(event) {
    const { name, value } = event.target;
    setConfig((current) => ({ ...current, [name]: Number(value) }));
  }

  async function startTraining(event) {
    event.preventDefault();
    setStarting(true);
    setError("");
    try {
      const created = await readRun(
        await fetch("/api/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config),
        }),
      );
      setRun(created);
    } catch (cause) {
      setError(cause.message);
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="Alien Garden home">
          <span className="brand-mark">✳</span>
          <span>GARDEN<span className="brand-light">LAB</span></span>
        </a>
        <div className="side-section-label">WORKSPACE</div>
        <button className="side-link selected"><span>◈</span> Training lab</button>
        <div className="sidebar-bottom">
          <div className="connection-indicator"><span className="pulse-dot" /> LOCAL SIMULATION</div>
          <div className="sidebar-version">MAPPO EXPERIMENTS <span>v0.1</span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">WORKSPACE <span>/</span> <b>TRAINING LAB</b></div>
          <div className={`status-pill ${status}`}>
            <span className="status-dot" />
            {status === "running" ? "TRAINING IN PROGRESS" : status === "completed" ? "RUN COMPLETE" : status === "failed" ? "RUN FAILED" : "READY TO TRAIN"}
          </div>
        </header>

        <section className="page-heading">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> COOPERATIVE MULTI-AGENT LEARNING</div>
            <h1>Alien Garden <span>Training Lab</span></h1>
            <p>Teach your alien gardeners to grow, harvest, and thrive together.</p>
          </div>
          {run && <div className="run-id">RUN <span>{run.id.slice(0, 8).toUpperCase()}</span></div>}
        </section>

        {error && <div className="error-banner" role="alert"><span>!</span>{error}<button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}
        {run?.error && <div className="error-banner" role="alert"><span>!</span>Training stopped: {run.error}</div>}

        <div className="dashboard-grid">
          <section className="panel config-panel">
            <div className="panel-heading">
              <div><div className="section-kicker">01 / CONFIGURATION</div><h2>Experiment setup</h2></div>
              <span className="panel-icon">⌘</span>
            </div>
            <form onSubmit={startTraining}>
              <div className="field-grid">
                <label className="field">
                  <span>Episodes</span>
                  <input name="episodes" type="number" min="1" max="500" value={config.episodes} onChange={updateConfig} disabled={status === "running" || starting} />
                </label>
                <label className="field">
                  <span>Random seed</span>
                  <input name="seed" type="number" value={config.seed} onChange={updateConfig} disabled={status === "running" || starting} />
                </label>
                <label className="field">
                  <span>Gardeners</span>
                  <input name="agents" type="number" min="1" max="8" value={config.agents} onChange={updateConfig} disabled={status === "running" || starting} />
                </label>
                <label className="field">
                  <span>Garden size</span>
                  <div className="input-suffix">
                    <input name="grid_size" type="number" min="3" max="24" value={config.grid_size} onChange={updateConfig} disabled={status === "running" || starting} />
                    <span>× {config.grid_size}</span>
                  </div>
                </label>
                <label className="field">
                  <span>Learning rate</span>
                  <input name="learning_rate" type="number" min="0.000001" max="0.1" step="any" value={config.learning_rate} onChange={updateConfig} disabled={status === "running" || starting} />
                </label>
                <label className="field">
                  <span>Steps / episode</span>
                  <input name="max_steps" type="number" min="1" max="1000" value={config.max_steps} onChange={updateConfig} disabled={status === "running" || starting} />
                </label>
              </div>
              <div className="config-footnote"><span>ⓘ</span> Runs are reproducible with the same seed and configuration.</div>
              <button className="start-button" type="submit" disabled={status === "running" || starting}>
                {status === "running" ? <><span className="button-spinner" /> Training in progress</> : starting ? "Starting run..." : <><span>▶</span> Start training run</>}
              </button>
            </form>
          </section>

          <section className="metrics-column">
            <div className="metrics-grid">
              <MetricCard label="EPISODES" value={`${String(episodes.length).padStart(2, "0")} / ${String(run?.config.episodes ?? config.episodes).padStart(2, "0")}`} note={status === "running" ? `${Math.round((run?.progress ?? 0) * 100)}% complete` : "training progress"} />
              <MetricCard label="MEAN CO-OP REWARD" value={averageReward.toFixed(2)} note="across completed episodes" accent="mint" />
              <MetricCard label="TOTAL HARVESTS" value={String(harvested).padStart(2, "0")} note="successful garden harvests" accent="gold" />
            </div>
            <section className="panel chart-panel">
              <div className="panel-heading chart-heading">
                <div><div className="section-kicker">02 / LEARNING CURVE</div><h2>Cooperative reward</h2></div>
                <span className="chart-legend"><i /> REWARD / EPISODE</span>
              </div>
              <RewardChart episodes={episodes} />
            </section>
          </section>
        </div>

        <section className="panel latest-panel">
          <div className="panel-heading latest-heading">
            <div><div className="section-kicker">03 / LIVE FEED</div><h2>Episode activity</h2></div>
            {run && <span className="feed-count">{episodes.length} {episodes.length === 1 ? "EPISODE" : "EPISODES"} LOGGED</span>}
          </div>
          {latest ? (
            <div className="activity-layout">
              <div className="latest-stat">
                <span className="latest-label">LATEST EPISODE</span>
                <strong>#{String(latest.episode).padStart(2, "0")}</strong>
                <span className="latest-reward">+{latest.cooperative_reward.toFixed(2)} <small>co-op reward</small></span>
              </div>
              <div className="activity-details">
                <div className="activity-measure"><span>Garden harvests</span><strong>{latest.harvested}</strong><div className="measure-track"><i style={{ width: `${Math.min(100, latest.harvested * 10)}%` }} /></div></div>
                <div className="activity-measure"><span>Seeds planted</span><strong>{latest.planted}</strong><div className="measure-track gold-track"><i style={{ width: `${Math.min(100, latest.planted * 10)}%` }} /></div></div>
                <div className="activity-measure"><span>Steps taken</span><strong>{latest.steps}</strong><small>of {run.config.max_steps} max</small></div>
              </div>
              <div className="agent-activity">
                <span className="latest-label">GARDENER REWARDS</span>
                <div className="agent-list">
                  {latest.agent_rewards.map((reward, index) => (
                    <div className="agent-chip" key={index}><span className="agent-avatar">{["✳", "✿", "❋", "✦", "❀", "✺", "✼", "✻"][index]}</span><span>AGENT {index + 1}</span><b>{reward.toFixed(2)}</b></div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="activity-empty"><span>✳</span><div><b>No episode data yet</b><small>Configure your experiment and start training to see gardener activity.</small></div></div>
          )}
        </section>
        <footer className="page-footer"><span>ALIEN GARDEN / MULTI-AGENT REINFORCEMENT LEARNING</span><span>POWERED BY <b>PYTORCH · MAPPO</b></span></footer>
      </main>
    </div>
  );
}

export default App;
