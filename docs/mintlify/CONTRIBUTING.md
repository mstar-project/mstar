# Mintlify documentation

This directory contains the M* Mintlify site. The existing
Sphinx source and publishing workflow remain active during migration.

## Preview and validate

Install Node.js 22 LTS, then run from this directory:

```bash
npx --yes mint@4.2.1184 dev
```

The preview opens at `http://localhost:3000`. Before submitting changes, run:

```bash
npx --yes mint@4.2.1184 validate
npx --yes mint@4.2.1184 broken-links
```

The pinned CLI is a documentation tool only; it is not a Python runtime dependency.

## Connect to Mintlify

1. Create a project at https://www.mintlify.com/start or open an existing project.
2. Connect the existing GitHub repository `mstar-project/mstar`. Install the Mintlify
   GitHub App with access to this repository. An organization owner or repository
   administrator may need to approve the installation.
3. In Git Settings, choose the production branch (`main` after these files merge).
4. Enable **docs.json is in a subdirectory** and enter `/docs/mintlify` without a
   trailing slash. Save the settings.
5. Check deployment status and open the site's URL from the Mintlify dashboard.
   The project name determines its `*.mintlify.site` URL; availability is checked
   during setup.
6. Add a custom domain in Mintlify when ready. Use the DNS records supplied by the
   dashboard, then verify the domain and HTTPS before changing public links.

Mintlify's GitHub App publishes updates from the configured branch. The Sphinx
workflow publishes the existing site independently.

Official instructions: [quickstart](https://www.mintlify.com/docs/quickstart),
[GitHub](https://www.mintlify.com/docs/deploy/github), and
[subdirectory setup](https://www.mintlify.com/docs/deploy/monorepo).

## Migration status

- Installation, quickstart, architecture, serving, clients, models, speech
  recognition, metrics, environment variables, evaluation, and model authoring
  are converted from the corresponding `.rst` guides. The introduction and
  `reference.mdx` provide the landing page and guide navigation.
- The generated Python API reference continues to use Sphinx. Migrating narrative
  pages does not replace the docstring generation pipeline. `api.mdx` links to
  that reference; Python symbol links in the guides open their source on GitHub.
- Both copies are maintained manually during migration. There is no automatic
  RST-to-MDX synchronization.

The `.rst` files remain the canonical sources during this transition, as required
by the repository's contribution and agent instructions. Update a guide's `.rst`
and `.mdx` files together when changing overlapping content. Check code blocks,
tables, callouts, internal links, and section anchors in the local preview.

Converted headings have explicit IDs, such as `## Metrics {#serving-metrics}`.
Keep an ID stable when editing a heading so incoming section links continue to
work. For links within Mintlify, use `/page` or `/page#section-id`.

Follow-ups are automated documentation checks, a decision on generated Python API
reference hosting, and a public-site cutover. At cutover, update the repository's
contribution and agent instructions to name the new canonical sources. Retire or
redirect the old site only after checking the replacement and its API-reference
coverage.
