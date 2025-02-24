# Viscore

### Setup

First, be sure to turn on the following within `Viscore` before trying to run the scripts:
- `REPL`
- `Cloud`

Then, before running any of scripts, be sure to install the following API / updates at the following address:

```bash
http://localhost:9090/jsd/repl.xhtml#local/update
```

Each update should be run within its own cell:

```bash
yield C.cloud_api("coralnet:install");
```

```bash
yield C.cloud_api("org3:install");
```

```bash
yield C.cloud_api("align:install");
```

If you have any issues, try closing `Viscore` and re-opening.


