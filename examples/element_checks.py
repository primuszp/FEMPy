"""A T3, T6 és Q4 elemek általános matematikai önellenőrzése."""

from primfem import verify_supported_elements

for report in verify_supported_elements(sample_count=50):
    print(report.summary())
    report.raise_for_failure()
