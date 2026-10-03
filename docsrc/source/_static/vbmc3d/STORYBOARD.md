# The PyVBMC video: storyboard

A narrated video of about two and a half minutes that explains what PyVBMC
does. `film.html` draws it in the wireframe style of `wordmark.html` and
plays that page's run (the banana, seed 42). The page itself stays an
unnarrated exhibit that a viewer can scrub and explore. The video walks
through the same run and introduces one idea at a time.

**Audience.** Scientists and engineers who fit models to data. They know
what a model, its parameters and a likelihood are. Some know Markov chain
Monte Carlo. Most have not met Gaussian processes, variational inference or
the ELBO. They are smart. The script explains ideas to them without talking
down.

**Angle.** Bayesian optimization finds the best-fitting parameters with few
evaluations. PyVBMC applies the same idea to the whole posterior. Viewers who
know Bayesian optimization recognize the idea at once. For the others the
comparison still makes sense, because everyone knows the difference between
a best fit and a best fit with error bars.

**Format.** 16:9, 1920 x 1080, 30 fps. A version with burned-in captions for
feeds that play muted, a clean version, and an `.srt`.

## Files

| File | Role |
|---|---|
| `film.html` | The page that draws the film. |
| `narration.json` | Every spoken line, its caption, and the rules that time each scene. |
| `film_timeline.js` | When each scene and line starts, written by `scripts/make_voice.py`. |
| `trace_intro.js` | The opening's emcee run and PyBADS run, written by `scripts/export_intro.py`. |

README.md ("The film") gives the commands that voice, record and mux the
film. Everything they produce apart from the two generated `.js` files
(voice takes and clips, sound, frames, videos) goes to the gitignored
`dev/media/vbmc3d-film/`.

The voice is Travis, a voice of the ElevenLabs Voice Library with a light
American accent, voiced with `eleven_multilingual_v2` one take per scene.
Kokoro (`af_heart`), an open model run locally, voiced the drafts while the
script changed, and `narration.json` can switch back to it.

## Style

- **Look.** The page's look, unchanged: dark ground, additive glowing wires,
  bloom and grain. The colours keep their meaning throughout. Evaluations
  are white, the surrogate cyan where it is sure and amber where it is not,
  the mixture magenta, the acquisition yellow and the true landscape grey
  (`:root` of `film.html`). In the opening the MCMC walkers are lavender and the
  PyBADS run orange.
- **Labels.** The narration uses no technical terms beyond posterior,
  likelihood, evidence and mixture of Gaussians. It explains each of them.
  The technical names (Gaussian process, variational inference, acquisition
  function, active learning) appear as small italic labels next to what they
  name, for viewers who want to look them up. A label stays at least 2.5 s.
- **Readouts.** The bottom left names the method on screen and counts its
  evaluations, with their cost at 3 min each. The same readout serves MCMC,
  Bayesian optimization and PyVBMC, so the costs compare at a glance. The
  number of mixture components and the evidence appear only once the
  narration has introduced them.
- **Words.** The unit is an *evaluation* of the likelihood, never a "run",
  which would be confused with a run of PyVBMC. Sentences are short and say
  one thing each. No sentence ends in a tacked-on clause that restates its
  first half, no colon introduces a punchline, and no appositive trails a
  sentence to round it off.
- **A nod.** Two dim readouts quote Doctor Who's "wibbly-wobbly,
  timey-wimey". While the surrogate is introduced, the top right reads
  `UNCERTAINTY wibbly-wobbly`. During the time-lapse, `timey-wimey` sits next
  to the cost readout. The video quotes the phrase and uses none of the
  BBC's imagery.
- **The footnotes.** Two asterisks refer to footnotes in the bottom right.
  The one on the MCMC count, in scene 2 and in the payoff, reads "\*Median of 100 runs:
  evaluations that emcee, a black-box MCMC sampler, needs to match the
  accuracy of a typical PyVBMC run on this posterior (mean marginal total
  variation). MCMC gives no evidence estimate."
  The one on line `w3` of scene 6 reads "\*Averaging the surrogate over
  the mixture has a closed form, Bayesian quadrature (O'Hagan 1991;
  Rasmussen & Ghahramani 2003, "Bayesian Monte Carlo")."

## Scenes

Times are those of Travis's takes and move whenever a scene is re-voiced.

### 1. The cost of a posterior (0:00–0:17)

| Line | Narration | Picture |
|---|---|---|
| p1 | Your model's likelihood takes minutes to compute. | The frame fades in from black on the floor grid, seen from low. One evaluation drops and lands. A label beside it says *3 MIN*. |
| p2 | You want the posterior over its parameters. | Labels on the floor edges, *parameter 1* and *parameter 2*. The camera begins a slow orbit. |
| p3 | The posterior tells you how plausible each setting is, given your data. | A crosshair crosses the floor with the label *HOW PLAUSIBLE?* |
| p4 | Picture the posterior as a landscape. | The true landscape rises out of the floor as a dim grey wireframe. |
| p5 | The higher the ground, the more plausible the setting. | Labels on the peak and on the flat between the arms, *MORE PLAUSIBLE* and *LESS PLAUSIBLE*. |

### 2. Markov chain Monte Carlo (0:17–0:26)

| Line | Narration | Picture |
|---|---|---|
| m1 | The textbook method is Markov chain Monte Carlo. | The 32 walkers of emcee, an MCMC sampler, appear in the plausible box one by one and move over the grey landscape, one evaluation at a time. The readout says *MCMC (EMCEE)*. |
| m2 | It needs tens of thousands of evaluations. | The ensemble speeds up until its dots cover the ridge. The count stops at *27,828\**. The footnote appears in the bottom right. |
| m3 | At a few minutes each, you would wait weeks. | The cost readout reaches *8.3 weeks*. |

### 3. Bayesian optimization (0:26–0:41)

| Line | Narration | Picture |
|---|---|---|
| o1 | Bayesian optimization finds the highest point of a landscape in far fewer evaluations. | The chain fades. The 67 points of a real PyBADS run appear one by one and close in on the peak. The readout says *BAYESIAN OPTIMIZATION (PYBADS)*. |
| o2 | It learns the landscape from the points it has seen, then decides where to explore next. | A faint line joins the points in order. A ring and the label *BEST FIT* mark the last one. |
| o3 | PyVBMC does the same for the whole posterior. | The true posterior glows magenta on the floor. The glow spreads from the best fit along the whole banana. |

### 4. In the dark (0:41–0:47)

| Line | Narration | Picture |
|---|---|---|
| s1 | We start in the dark. | The grey landscape, the points and the glow fade out. The readout switches to *PYVBMC* at zero evaluations. |
| s2 | Each evaluation reveals the height at a single point. | The ten evaluations of the initial design drop one by one. |

### 5. The surrogate (0:47–1:04)

| Line | Narration | Picture |
|---|---|---|
| g1 | From these few points, PyVBMC builds a surrogate. | The first surrogate grows out of the nodes. |
| g2 | The surrogate is a cheap statistical model of the whole landscape. | Label *Gaussian process* above the highest node. |
| g3 | At the evaluated points it holds still. | The camera closes in on that node. The wires through it do not move. Label *EXACT HERE*. |
| g4 | Everywhere else it wobbles. | The camera pulls back over the outskirts. Readout `UNCERTAINTY wibbly-wobbly`. |
| g5 | The more it wobbles, the less it knows. | The colour key appears: amber unsure, cyan sure. |

### 6. The mixture (1:04–1:25)

| Line | Narration | Picture |
|---|---|---|
| q1 | On the surrogate, PyVBMC fits a mixture of Gaussians. | The mixture's two components appear on the floor as ellipses. Its magenta sheet rises over them. Label *variational inference*. The component count appears in the readout. |
| w1 | Why not use the surrogate itself? | Hold on the surrogate and the mixture. |
| w2 | It gives a height at every point, but it's not directly usable. | The camera keeps orbiting. |
| w3 | A mixture of Gaussians you can sample and compute with, exactly.\* | Eighty samples of the mixture fall onto the floor, one after another, and land inside its ellipses. The Bayesian quadrature footnote appears in the bottom right. |
| q2 | This is its first guess at the posterior. | The magenta sheet beside the surrogate's ridge. |
| q3 | For now it is a crude one. | Hold. The two components are visibly blunter than the ridge. The samples and the footnote fade with the scene. |

### 7. Where to evaluate next (1:25–1:40)

| Line | Narration | Picture |
|---|---|---|
| a1 | Where to evaluate next? | The camera rises. The acquisition heat fades in on the floor. Label *acquisition function*. |
| a2 | An optimizer would head for the peak. | PyBADS's points from scene 3 flicker at the peak, with the label *OPTIMIZER*. |
| a3 | PyVBMC is after the whole posterior, so it explores places that are plausible and still uncertain. | A yellow beam marks the brightest spot. The first evaluation of iteration 1 drops into it. The other four follow, each into its own heat. The surrogate and the mixture refit. The label *active learning* appears on the beam over the first evaluation. |

### 8. The loop (1:40–1:45)

| Line | Narration | Picture |
|---|---|---|
| l1 | Evaluate. | Iteration 2. A three-step indicator appears top right and lights *EVALUATE* as its five points land. |
| l2 | Update the surrogate. | It lights *SURROGATE* as the surrogate refits. |
| l3 | Refit the mixture. | It lights *MIXTURE* as the mixture refits. |

### 9. Time-lapse (1:45–2:03)

| Line | Narration | Picture |
|---|---|---|
| — | | Iterations 3 to 19, each shorter than the last, then the final boost. The indicator keeps cycling. `timey-wimey` sits beside the cost readout. |
| l4 | Round after round, the wobbling dies down. | The sheet turns from amber to cyan. The `wibbly-wobbly` readout fades. |
| l5 | The mixture grows new components to follow the ridge. | The component count climbs from 2 to 21, then to 50 at the final boost. |

### 10. The reveal (2:03–2:09)

| Line | Narration | Picture |
|---|---|---|
| r1 | PyVBMC never saw the true landscape. | The camera orbits. |
| r2 | Here it is. | The grey landscape rises into place under the surrogate and the mixture. Where they coincide, the wires add up to white. |

### 11. The evidence (2:09–2:18)

| Line | Narration | Picture |
|---|---|---|
| e1 | The same calculation gives an estimate of the model's evidence. | The view morphs to density. The readout adds *LOG EVIDENCE −0.019*. |
| e2 | You need that number to compare two models. | Hold. |
| e3 | The estimate lands close to the true value. | *TRUE +0.000* appears beside it. |

### 12. The payoff (2:18–2:28)

| Line | Narration | Picture |
|---|---|---|
| f1 | All this from a hundred evaluations. | The scene dims. *PYVBMC 105 EVALUATIONS* appears in the centre. |
| f2 | Markov chain Monte Carlo would have needed tens of thousands. | Beside it, *MCMC 27,828\* EVALUATIONS*, the budget of scene 2. The footnote returns. |
| f3 | Weeks of waiting became an afternoon. | Under each count, its cost, *5 h 15 min* and *8.3 weeks*, then *AT 3 MIN PER EVALUATION · A 2-PARAMETER EXAMPLE*. |

### 13. The wordmark (2:28–2:35)

No narration. This is the page's finale. The camera pulls up until the
posterior is the V of the wordmark. "Py" and "BMC" appear beside it. Then
*1.5* appears after the word, smaller and in cyan, its top on the letters'
cap line, as the version hangs from the top of `logo.svg`.

### 14. End card (2:35–2:43)

No narration. The wordmark dims behind **PyVBMC 1.5** and the method's
name, Variational Bayesian Monte Carlo, then `pip install pyvbmc` and
acerbilab.org/model-fitting, the lab's page of model-fitting tools, which
links PyVBMC and PyBADS. Below them are the three references of the
README (Huggins et al. 2023, JOSS; Acerbi 2018, NeurIPS; Acerbi 2020,
NeurIPS), then three credit lines: "Machine and Human Intelligence Group ·
University of Helsinki", "Research Council of Finland · ELLIS Institute
Finland" and "Directed by Luigi Acerbi · Made with Claude Code · Voice:
ElevenLabs".

## Sound

`scripts/make_score.py` synthesizes the score from the times at which the
page shows things. Its harmony stays in D minor through the cost of a
posterior and the run, and turns to D major on "Here it is" and again for
the wordmark. The evaluations on screen are heard. The MCMC run clicks
once per evaluation, and its clicks thicken into a roar that stops when
the run does. PyBADS's points are warm pings, and PyVBMC's are glassy
pings, higher for higher ground. A high fifth wobbles in pitch while the
surrogate is unsure and steadies during the time-lapse. The loop's three
steps play the chord's tones upwards, so the time-lapse turns them into an
accelerating arpeggio. The score ducks under the voice. `scripts/mux.py`
sets the loudness to -16 LUFS.

## Claims to check

- *Tens of thousands of evaluations* for MCMC. The matched-budget
  measurement (`dev/results/2026-09-30-matched-mcmc-budget.md` on
  `dev-next`) finds that emcee, the fastest of three tuned black-box
  samplers, needs a median of 27,828 evaluations (90% interval 23,098 to
  31,764, from 100 runs each) to match the MMTV of a typical PyVBMC run,
  which uses 85. Scene 2 shows one emcee run with the same setting, stopped
  at that budget, and the footnote gives the definition.
- *A hundred evaluations* against that budget. The payoff sets the film's
  run, 105 evaluations, beside a budget matched to a typical run, which
  uses 85, so the comparison does not flatter PyVBMC.
- *Far fewer evaluations* for Bayesian optimization. The PyBADS run of
  scene 3 reaches the peak in 67 evaluations.
- *Weeks* and *an afternoon* assume 3 min per evaluation, as the readout
  says. 27,828 evaluations take 8.3 weeks and 105 take 5 h 15 min.
  PyVBMC's own computing, about a minute on this problem, is not counted.
- *A hundred evaluations* holds for this two-parameter example. PyVBMC
  needs more evaluations as the number of parameters grows, hence the label.
- *The estimate lands close to the true value*. The ELBO is −0.019 and the
  log evidence of the target is 0.
- *A mixture of Gaussians you can compute with, exactly*. The surrogate is
  a Gaussian process with a squared-exponential kernel and a
  negative-quadratic mean, so its average under a mixture of Gaussians, the
  expected log joint, has a closed form (`_gp_log_joint` in
  `pyvbmc/vbmc/variational_optimization.py`). The mixture's entropy, the
  other term of the ELBO, has none, and PyVBMC estimates it (`entropy/`).
- *The same calculation gives an estimate of the model's evidence*. The
  ELBO, PyVBMC's estimate of the log evidence, is that expected log joint
  plus the entropy.
- *Holds still at the evaluated points*. The evaluations are exact, so the
  surrogate's uncertainty is zero there (README, "What the sheet is").
