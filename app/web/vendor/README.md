# vendor — 同梱するフロントエンドライブラリ

| ファイル | 版 | 由来 | ライセンス |
|---|---|---|---|
| `smiles-drawer.min.js` | 2.4.1 | [reymond-group/smilesDrawer](https://github.com/reymond-group/smilesDrawer)（npm `smiles-drawer`） | MIT（`LICENSE-smiles-drawer.md`） |

**なぜ同梱するか** — sandbox / 本番環境ではネットワークを使えないことがあり、
生成した HTML レポートはオフラインでも構造が描画できる必要があるため、CDN を参照せず
リポジトリに置いています。Web UI は `/static/smiles-drawer.min.js` として配信し、
HTML レポート（`render_report_html` ツール）はこのファイルを**インライン埋め込み**します。

**更新手順**

```bash
curl -sL -o app/web/vendor/smiles-drawer.min.js \
    https://unpkg.com/smiles-drawer@<version>/dist/smiles-drawer.min.js
# ライセンス本文も同じ版のものへ更新する（npm tarball の LICENSE.md）
```

更新したら `pytest tests/test_report.py tests/test_api.py` で、描画 API
（`window.SmiDrawer` / `data-smiles` 属性 / `>>` を含む反応 SMILES）が
そのまま使えることを確認してください。

**使っている API**（2.4.1）

- `SmiDrawer.apply(moleculeOptions, reactionOptions, attribute, theme)` — `[data-smiles]` を一括描画
- `new SmiDrawer(moleculeOptions, reactionOptions).draw(smiles, svgElement, theme, successCb, errorCb)`
- `data-smiles` に `>` を含む文字列を渡すと反応式として描画される（例 `CCO.CC(=O)O>>CCOC(C)=O`）
- 属性値に空白を入れないこと（空白以降は反応オプション/試薬テキストとして解釈される）
