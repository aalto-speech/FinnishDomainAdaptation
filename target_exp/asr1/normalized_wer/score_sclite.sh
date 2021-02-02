#!/bin/bash -e

test_set="${1}"
shift

references="${test_set}/normalized.trn"
echo refs "${references}"

references_8bit=$(mktemp ${TMPDIR}/$(basename "$0").XXXXXXXXXX)
iconv -f UTF-8 -t ISO-8859-15 <${references} >"${references_8bit}"

echo params "${@}"

for hypothesis in "${@}"
do
	if [ ! -e "${hypothesis}" ]
	then
		echo "Hypothesis file does not exist: ${hypothesis}" 2>&1
		exit 1
	fi

	hypothesis_8bit=$(mktemp --tmpdir $(basename "${0}").XXXXXXXXXX)
	iconv -f UTF-8 -t ISO-8859-15 <"${hypothesis}" >"${hypothesis_8bit}"

	echo $(basename "${hypothesis}" .trn)
	sclite -o sum -o stdout -i wsj -f 0 -h "${hypothesis_8bit}" -r "${references_8bit}" | grep 'Sum'
	rm -f "${hypothesis_8bit}"
done

rm -f "${references_8bit}"
