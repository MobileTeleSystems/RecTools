from datetime import date
import logging
from pathlib import Path
import sys
import click
import typing as tp
import re
import git

FILE_PATTERN = "*.py"

ROOT_PATH = Path(__file__).parents[1].absolute()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rectools")
logger.setLevel(logging.INFO)


LICENSE_KEY_PHRASES = ( 
    "licensed under",
    "copyright",
    "without warranties or conditions of any kind"
)

LISCENSE_HEADER_REGEX = re.compile(
r"""#  Copyright .{4,30} MTS \(Mobile Telesystems\)
#
#  Licensed under the Apache License, Version 2.0 \(the "License"\);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
""",
 flags=re.MULTILINE)


LISCENSE_HEADER_TEMPLATE = """#  Copyright {years} MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""


def collect_files(sources: tp.List[str]) -> tp.List[Path]:
    files = []
    for source in sources:
        path = Path(source)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for file_path in path.rglob(FILE_PATTERN):
                files.append(file_path)
        else:
            raise ValueError(f"Invalid source (neither a file nor a directory): {source}")
    return files

def has_license(content: str) -> bool:
    return all(phrase in content.lower() for phrase in LICENSE_KEY_PHRASES)


def extract_header(content: str) -> tp.Optional[str]:
    m = re.match(LISCENSE_HEADER_REGEX, content)
    if m is not None:
        return m.group()
    return None


def remove_header(content: str) -> str:
    return re.sub(LISCENSE_HEADER_REGEX, "", content)


def get_file_first_and_last_modification_years(file_path: Path) -> tp.Tuple[int, int]:
    repo = git.Repo(ROOT_PATH)
    commits = list(repo.iter_commits(paths=file_path))
    if commits:
        return (date.fromtimestamp(commit.committed_date).year for commit in (commits[-1], commits[0]))
    
    # No commits means file has just been added
    current_year = date.today().year
    return current_year, current_year


def make_header(file_path: str) -> str:
    creation_year, last_modification_year = get_file_first_and_last_modification_years(file_path)
    if creation_year == last_modification_year:
        years_str = str(creation_year)
    else:
        years_str = f"{creation_year}-{last_modification_year}"
    header = LISCENSE_HEADER_TEMPLATE.format(years=years_str)
    return header


def add_header(content: str, header: str) -> str:
    if content:
        header += ""  ###
    return header + content


def replace_header(content: str, new_header: str) -> str:
    fixed_content = remove_header(content)
    fixed_content = add_header(content, new_header)
    return fixed_content


def process_file(file_path: Path, check: bool) -> bool:
    content = file_path.read_text()
    correct_header = make_header(file_path)
    
    if not has_license(content):
        logger.info(f"{file_path}: doesn't have a license header")
        if check:
            return False
        else:
            fixed_content = add_header(content, correct_header)
            file_path.write_text(fixed_content)
            logger.info(f"{file_path}: fixed")
            return True
        
    current_header = extract_header(content)
    if current_header is None:
        logger.error(f"{file_path}: file have a license info, but header cannot be extracted. Manual fix required.")
        return False
    
    if current_header != correct_header:
        logger.info(f"{file_path}: incorrect license header")
        if check:
            return False
        else:
            fixed_content = replace_header(content, correct_header)
            file_path.write_text(fixed_content)
            logger.info(f"{file_path}: fixed")


@click.command()
@click.option("--check", is_flag=True, default=False)
@click.argument("sources", type=click.STRING, nargs=-1)
def cli(sources: tp.List[str], check: bool) -> None:
    any_error = False
    file_paths = collect_files(sources)

    for file_path in file_paths:
        processed_fine = process_file(file_path, check)
        if not processed_fine:
            any_error = True
    
    if any_error:
        logger.error("Some files have incorrect license headers")
        sys.exit(1)
    else:
        if check:
            logger.info("Everything is fine!")
        else:
            logger.info("All problems are fixed!")
        sys.exit(0)


def main() -> None:
    return cli()


if __name__ == "__main__":
    main()