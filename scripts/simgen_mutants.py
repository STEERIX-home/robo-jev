"""Task v0a 리뷰 1 I1·v0b — 변이 시험: 사후·굴리기·정답·질문을 일부러 망가뜨린 사본에서 simgen 시험이 떨어지는지 본다.

    uv run python scripts/simgen_mutants.py --work artifacts/scratch/v0a/mutants --out artifacts/scratch/v0a/mutants/result.json

리뷰가 만든 변이체 넷(M1 추적 외삽 부호 뒤집기, M2 의도 사후를 사전으로, M3 채움 사후를 구간 가운데의 가우스로, M4 마주 오는 상대가
감속하지 않음)과 같은 뜻의 변경을 지금 코드에 넣는다(줄 번호는 바뀌었다 — 바꿀 문장을 글자 그대로 찾고, 정확히 한 번 나와야 한다).
Task v0b는 v0a 재검증이 시험 없이 남긴 수정 경로(m-a 셋)·m-b·m-c·끼어듦 전제와 새 현상(P3·P4·P6)의 변이체를 더한다(``--only``로 고른다).
변이마다 저장소의 src·tests·pyproject를 작업 폴더에 복사하고, 그 사본의 src를 PYTHONPATH 앞에 두어 simgen 시험 전체를 돌려 떨어진 시험의
이름을 모은다. 변이 없는 사본(control)은 모두 통과해야 한다. GPU·모델을 쓰지 않는다(systemd 유닛 안에서 돌린다).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 이름 → (파일, 원래 문장, 바꾼 문장, 설명) 또는 (설명, [(파일, 원래, 바꾼) …]) — 여러 곳을 함께 바꾸는 변이체
MUTANTS = {
    "M1": ("src/robo_jev/simgen/posterior.py", "now = last + direction * (speed + offset) * agent.age", "now = last - direction * (speed + offset) * agent.age",
           "track extrapolation sign flipped"),
    "M2": ("src/robo_jev/simgen/posterior.py", "return (1.0 - q) * given_cue + q * prior", "return prior", "intent posterior replaced by the prior"),
    "M3": ("src/robo_jev/simgen/posterior.py", "fill = _fill_posterior(rng, cfg, c, k)",
           "fill = np.clip(c.fill.value + c.fill.sigma_eff * rng.standard_normal(k), 0.02, 1.0)", "fill posterior = Gaussian at the bin centre"),
    "M4": ("src/robo_jev/simgen/rollout.py", 'trigger = (b["reactive"] > 0.5) & overlap', 'trigger = (b["reactive"] > 99.0) & overlap',
           "head-on agents never brake"),
    # v0b — v0a 재검증의 시험 없는 수정 경로(m-a)·m-b·m-c·끼어듦 전제
    "Ma1": ("src/robo_jev/simgen/scenes.py", 'coord_now, rate = v["dist"] - r_sum, v["va"] + r.speed', 'coord_now, rate = v["dist"] - r_sum, v["va"]',
            "head-on observation side reverted (fix = now + agent speed only)"),
    "Ma2": ("src/robo_jev/simgen/generate.py", "    exact = exact_labels(cfg, obs)\n", "    exact = {}\n", "closed-form calibration of the batch switched off"),
    "Ma3": ("src/robo_jev/simgen/scenes.py", "    if age > age_max:\n", "    if False:\n", "step-in last-fix cap removed"),
    "Mb": ("src/robo_jev/simgen/rollout.py", "np.array(contact) | np.array(side_contact)", "np.array(contact)", "side-obstacle hits do not spill"),
    "Mc": ("duplicate questions allowed", [
        ("src/robo_jev/simgen/questions.py", "        return [cand for cand in cands if cand.ident not in used]", "        return list(cands)"),
        ("src/robo_jev/simgen/questions.py", "        if identity(second.recipe) != identity(primary.recipe):", "        if True:"),
        ("src/robo_jev/simgen/questions.py", "        assert identity(q.recipe) == cand.ident, (cand.ident, q.recipe)\n", ""),
    ]),
    "Mpremise": ("src/robo_jev/simgen/posterior.py", "    ok = np.flatnonzero((entering >= 0.0) & (entering <= band * (obs.robot.radius + ra)))",
                 "    ok = np.arange(m)", "step-in posterior ignores the text premise"),
    # v0b — 새 현상·빼 둔 칸·교란·한국어
    "Mp3prior": ("src/robo_jev/simgen/p3.py", '"com": _uniform(rng, spec["com_ratio"], k)', '"com": _uniform(rng, (spec["com_ratio"][0], spec["com_ratio"][0] + 0.02), k)',
                 "P3 posterior: centre-of-mass ratio squeezed to the bottom of its range"),
    "Mp4prior": ("src/robo_jev/simgen/p4.py", "return StateModel(family, dims, prior / prior.sum(), mean, sd, np.array([0, 1, 2, 2]))",
                 "return StateModel(family, dims, np.full(4, 0.25), mean, sd, np.array([0, 1, 2, 2]))", "P4 state prior replaced by a uniform one"),
    "Mp6stay": ("src/robo_jev/simgen/p6.py", "    return _uniform_exp_mean(lo, hi, age)\n", "    return math.exp(-0.5 * (lo + hi) * age)\n",
                "P6 still-there probability at the mean rate (Jensen error)"),
    "Mvoi": ("src/robo_jev/simgen/voi.py", "expected = (self.likelihood * w[None, :]) @ costs.T", "expected = self.likelihood @ costs.T",
             "look-again decision ignores the posterior weights"),
    "Mstack": ("src/robo_jev/simgen/p3.py", "toward = sum(w * share(lo, hi) for lo, hi, w in pieces)", "toward = sum(w for lo, hi, w in pieces if (lo >= 0) == (st.offset > 0))",
               "stack weight-side closed form back to the piece-sign bug"),
    "Mcorner": ("src/robo_jev/simgen/p6.py", "            d_hard = ph.stopping_distance(obs.robot.speed, obs.robot.latency, obs.robot.decel_hard)\n",
                "            d_hard = ph.stopping_distance(obs.robot.speed, obs.robot.latency, obs.robot.decel_moderate)\n",
                "blind-corner closed form uses firm instead of maximum braking"),
    "Mholdout": ("training split may hit range cells", [
        ("src/robo_jev/simgen/holdout.py", "    if split.train_ranges and split.cell is None and hit:\n", "    if False:\n"),
        ("src/robo_jev/simgen/holdout.py", "        elif split.train_ranges:\n", "        elif False:\n"),
    ]),
    "Mmirror": ("src/robo_jev/simgen/perturb.py", '        v["lat0"], v["dir"] = -v["lat0"], -v["dir"]\n', '        v["lat0"] = -v["lat0"]\n',
                "T6 mirror forgets the step-in direction"),
    "Mjosa": ("src/robo_jev/simgen/korean.py", "    return with_final if final else without\n", "    return with_final\n", "Korean particles ignore the final sound"),
    "Mimpact": ("src/robo_jev/simgen/questions.py", '    return obs.scene == "p3p5_carrying_mixed" and obs.env.maneuver == "brake"\n', "    return False\n",
                "spill wording for P3+P5 braking stops naming the collision"),
    # 리뷰 1(v0b)의 살아남은 변이체 넷 — 같은 변경(수정 1 M3)
    "MR1": ("src/robo_jev/simgen/perturb.py", '    return _DECIMAL.sub(lambda m: m.group(1) + ("." + m.group(2) if m.group(2) else ""), text)',
            "    return _DECIMAL.sub(lambda m: m.group(1), text)", "T6 number notation drops all decimals (values change)"),
    "MR2": ("src/robo_jev/simgen/narrate_ko.py", '            text = f"라이다는 약 {dist} 앞 {where}에 물체가 있다고 보고하지만, 카메라에는 그곳에 아무것도 보이지 않습니다."',
            '            text = f"카메라는 약 {dist} 앞 {where}에 물체가 있다고 보고하지만, 라이다는 그곳에서 반사를 받지 못합니다."',
            "Korean sensor-conflict sentence names the wrong sensor"),
    "MR4": ("src/robo_jev/simgen/p4.py", "            means[i, j], sds[i, j] = post_mean, math.sqrt(post_var + se**2)", "            means[i, j], sds[i, j] = post_mean, se",
            "P4 re-measurement predictive ignores the posterior uncertainty"),
    "MR6": ("src/robo_jev/simgen/p6.py", '        p_seen = np.where(still, look["detect"], look["false_alarm"])', '        p_seen = np.where(still, look["detect"], look["detect"])',
            "stale-track look made uninformative"),
    # 수정 1의 고침 — 되돌리면 떨어져야 한다
    "Mrobotcut": ("I1 reverted: range cells only in their listed scenes", [
        ("src/robo_jev/simgen/holdout.py", "        if value is not None and cell.in_test(value, body) and holdout.reported_under(cell, scene):",
         "        if value is not None and cell.in_test(value, body) and holdout.reported_under(cell, scene) and scene in cell.scenes:"),
        ("src/robo_jev/simgen/holdout.py", "        elif split.train_ranges:\n            narrow(cell.var, cell.train_bounds(body))",
         "        elif split.train_ranges and scene in cell.scenes:\n            narrow(cell.var, cell.train_bounds(body))"),
    ]),
    "Mfallimpact": ("src/robo_jev/simgen/questions.py", '    return obs.scene in ("p3_carrying_tall", "p3p5_carrying_mixed") and obs.env.maneuver == "brake"\n',
                    "    return False\n", "I2 reverted (text): P3 braking fall questions stop naming the collision / motion-only qualifier"),
    "Mfallmotion": ("src/robo_jev/simgen/p3.py", "        motion.append(t_l | t_s | off_l | off_s)", "        motion.append(t_l | t_s | off_l | off_s | hit)",
                    "I2 reverted (label): the motion-only fall counts the collision"),
    "Mbrakekey": ("src/robo_jev/simgen/p3.py", '    return {"brake": "stop_hard", "turn": "turn_100", "start": "start_quick"}[maneuver]',
                  '    return {"brake": "stop_moderate", "turn": "turn_100", "start": "start_quick"}[maneuver]', "I2: braking key back to the firm stop"),
    "Mstatic": ("I3 reverted: a stationary robot is walked into as often as a moving one", [
        ("configs/simgen/catalog-v0.2.yaml", "  fail: {attentive: 0.01, distracted: 0.05}", "  fail: {attentive: 0.25, distracted: 0.25}"),
        ("configs/simgen/catalog-v0.2.yaml", "  fail_close: {attentive: 0.05, distracted: 0.50}", "  fail_close: {attentive: 0.25, distracted: 0.25}"),
    ]),
    "Mstopbefore": ("src/robo_jev/simgen/p6.py", "    room = ahead_reported - r.speed * r.latency - _BEFORE_BUFFER", "    room = ahead_reported + 0.3",
                    "I3: stop-before-the-corner forgets the reaction distance and buffer"),
    "Mpairkeep": ("src/robo_jev/simgen/generate.py", '            cand = assemble(cfg, moment, j, id_prefix=id_prefix, t6={"recommendation": keep}, threshold_obs=base.obs, **common)',
                  "            cand = assemble(cfg, moment, j, id_prefix=id_prefix, threshold_obs=base.obs, **common)", "I4: pair variants recompute the recommendation"),
    "Mpaircross": ("src/robo_jev/simgen/generate.py", "            if p1 is not None and (p0 - 0.5) * (p1 - 0.5) < 0:", "            if p1 is not None:",
                   "I4: pairs kept even when the key label does not cross 0.5"),
    "Mthreshold": ("src/robo_jev/simgen/questions.py", "    c.tobs = threshold_obs or obs", "    c.tobs = obs", "I4: pair variants place rule thresholds at their own values"),
    "Mhedge": ("src/robo_jev/simgen/narrate.py", 'HEDGED_CUES = ("glass", "black")', "HEDGED_CUES = ()", "M2 reverted: glass/black cues asserted as objects"),
    "Mofferbefore": ("src/robo_jev/simgen/p6.py", '    if obs.scene == "p6_blind_corner" and not stop_before_feasible(obs):\n', "    if False:\n",
                     "N1 reverted: stop-before-the-corner offered even when maximum braking cannot do it"),
    "Mpairskip": ("src/robo_jev/simgen/generate.py", '        "pair_skip": moment.pair_skip,\n', '        "pair_skip": None if moment.pair_skip is None else 0,\n',
                  "N2 reverted: the accepted numeric-pair candidate is not recorded"),
    "Midentical": ("src/robo_jev/simgen/testsets.py",
                   '    record["meta"]["t6"]["identical_to_base"] = json.dumps(record["request"], ensure_ascii=False) == json.dumps(base["request"], ensure_ascii=False)',
                   '    record["meta"]["t6"]["identical_to_base"] = False', "M4: identical T6 variants not flagged"),
}  # fmt: skip


def _edits(name: str) -> tuple[str, list[tuple[str, str, str]]]:
    spec = MUTANTS[name]
    if len(spec) == 2:
        return spec[0], list(spec[1])
    path, old, new, description = spec
    return description, [(path, old, new)]
_FAILED = re.compile(r"^FAILED (\S+?)(?: - .*)?$")


def _copy(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(REPO / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "tests", dest / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "pyproject.toml", dest / "pyproject.toml")
    configs = REPO / "configs" / "simgen"
    shutil.copytree(configs, dest / "configs" / "simgen")


def _mutate(dest: Path, path: str, old: str, new: str) -> None:  # noqa: D401
    target = dest / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one '{old}', found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def _run(dest: Path, basetemp: Path) -> tuple[int, list[str], str]:
    tests = sorted(str(p.relative_to(dest)) for p in (dest / "tests").glob("test_simgen_*.py"))
    env = {**os.environ, "PYTHONPATH": str(dest / "src"), "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", "-p", "no:randomly", f"--basetemp={basetemp}", *tests]
    done = subprocess.run(cmd, cwd=dest, env=env, capture_output=True, text=True, timeout=3600)
    failed = [m.group(1) for line in done.stdout.splitlines() if (m := _FAILED.match(line.strip()))]
    summary = next((line for line in reversed(done.stdout.splitlines()) if " passed" in line or " failed" in line), "")
    return done.returncode, failed, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True, help="사본을 둘 작업 폴더(저장소 밖, 예: artifacts/scratch/…)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="이 변이체만(기본: 전부)")
    parser.add_argument("--jobs", type=int, default=1, help="동시에 돌릴 사본 수")
    args = parser.parse_args(argv)
    results = {}
    names = [n for n in MUTANTS if args.only is None or n in args.only]

    def one(name: str) -> tuple[str, dict]:
        dest = args.work.resolve() / name
        _copy(dest)
        if name != "control":
            for path, old, new in _edits(name)[1]:
                _mutate(dest, path, old, new)
        code, failed, summary = _run(dest, args.work.resolve() / f"pytest-{name}")
        # 사본이 정말 쓰였는지: 사본의 패키지 경로를 확인한다
        where = subprocess.run([sys.executable, "-c", "import robo_jev.simgen as s; print(s.__file__)"], cwd=dest,
                               env={**os.environ, "PYTHONPATH": str(dest / "src")}, capture_output=True, text=True).stdout.strip()  # fmt: skip
        print(f"{name}: {summary} ({len(failed)} failed) — {where}", flush=True)
        return name, {"description": _edits(name)[0] if name in MUTANTS else "no mutation", "returncode": code, "summary": summary,
                      "failed": failed, "package": where}  # fmt: skip

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for name, result in pool.map(one, ("control", *names)):
            results[name] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = results["control"]["returncode"] == 0 and all(results[m]["failed"] for m in names)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
