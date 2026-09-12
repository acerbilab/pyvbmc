# 2026-09-02 — A PyVBMC skill for users' coding agents

**Status:** thin documentation wrapper implemented 2026-09-12 at
[`skills/pyvbmc/SKILL.md`](../skills/pyvbmc/SKILL.md), with a discovery and
copying note in the main README.

## Purpose and scope

Help a user's coding agent find the relevant PyVBMC documentation when
setting up an analysis, troubleshooting a run or interpreting results.
The skill routes tasks to the README, quickstart, FAQ, existing examples
and API reference. Scientific explanations and API guidance remain in
those documents.

The PI selected this thin first version on 2026-09-12. It replaces the
broader proposal for separate reference material, a worked example,
helpers and package installation support. The skill adds only reading
directions, version checks and instructions to account for model evaluations
within the user's budget and inspect existing results before another run.

## Distribution and maintenance

The single source is `skills/pyvbmc/SKILL.md`. Users can point an agent to
that file or copy its enclosing folder to their agent's skill directory.
The wrapper requires access to the documentation through a checkout or
the web. Repository links are absolute so a copied folder can resolve them.
There is no installer or wheel integration in this scope.

The skill accompanies PyVBMC 1.5 and directs agents to check the installed
version. Its source links use `dev-next` while the 1.5 documentation awaits
publication. At the pre-release documentation review, update these links
to the release documentation and verify them. Copied skills are updated by
copying the folder from the relevant PyVBMC version again.

Expand the wrapper when use reveals a concrete gap in how agents find or
apply the documentation. Add general scientific guidance to the maintained
docs and link to it from the skill.
