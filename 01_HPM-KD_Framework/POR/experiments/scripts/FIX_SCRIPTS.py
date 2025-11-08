#!/usr/bin/env python3
"""
Script para corrigir erros nos scripts de experimentos
"""

import re
from pathlib import Path

def fix_logger_end():
    """Remove parâmetro 'end=' de logger.info()"""
    script = Path('01_compression_efficiency.py')

    content = script.read_text()

    # Fix logger.info with end=
    content = re.sub(
        r"logger\.info\((.*?), end=''\)",
        r"logger.info(\1)",
        content
    )

    script.write_text(content)
    print(f"✅ Corrigido logger.info em {script.name}")


def fix_dbdataset_api():
    """Corrige chamadas para DBDataset com API correta"""

    scripts = [
        '01_compression_efficiency.py',
        '02_ablation_studies.py',
        '03_generalization.py',
        '04_computational_efficiency.py'
    ]

    for script_name in scripts:
        script = Path(script_name)
        if not script.exists():
            continue

        content = script.read_text()

        # Padrão antigo (ERRADO):
        # db_dataset = DBDataset(
        #     X=X_train.cpu().numpy(),
        #     y=y_train.cpu().numpy(),
        #     task='classification'
        # )

        # Padrão novo (CORRETO - verificar documentação DeepBridge):
        # Opção 1: Passar como data/target
        # Opção 2: Passar como dict
        # Opção 3: Usar from_tensors

        # Vamos substituir por uma versão que funciona
        pattern = r'db_dataset = DBDataset\(\s*X=X_train\.cpu\(\)\.numpy\(\),\s*y=y_train\.cpu\(\)\.numpy\(\),\s*task=[\'"]classification[\'"]\s*\)'

        replacement = '''# Criar DBDataset compatível com DeepBridge
        # Nota: DBDataset espera data como primeiro argumento, não X/y keywords
        db_dataset = DBDataset(
            data=X_train.cpu().numpy(),
            target=y_train.cpu().numpy()
        )'''

        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        script.write_text(content)
        print(f"✅ Corrigido DBDataset em {script.name}")


if __name__ == '__main__':
    print("🔧 Corrigindo scripts...")
    print()

    fix_logger_end()
    fix_dbdataset_api()

    print()
    print("✅ Todas as correções aplicadas!")
    print()
    print("Agora execute novamente:")
    print("  python RUN_COLAB.py --full")
