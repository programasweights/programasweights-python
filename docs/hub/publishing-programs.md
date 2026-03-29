# Publishing Programs

## When a program becomes public

Programs are **published automatically** when you compile on the website. New programs are **public by default**, so they appear on the hub and can be discovered by others unless your workflow changes that behavior.

## Naming a program

To assign a memorable name:

1. Sign in with **GitHub**.
2. Compile your specification on the website.
3. Enter a name in the **“Name this program”** field when prompted.

Your program receives a slug of the form:

```text
your-github-username/your-name
```

Replace `your-github-username` with your GitHub handle and `your-name` with the label you chose.

## Official programs

Programs under the `programasweights/` namespace are **official** releases. They are curated and maintained by the PAW team rather than individual users.

## Content addressing and aliases

Programs are **content-addressable**: compiling the **exact same** specification yields the **same program ID**, regardless of who runs the compile.

Different people can still attach **different aliases** to that same underlying program. Multiple hub entries may therefore refer to one canonical compiled artifact while exposing distinct names for discovery and documentation.
