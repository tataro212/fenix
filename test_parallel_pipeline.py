import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from models import PageModel, ProcessResult
from config_manager import Config
from processing_strategies import process_page_worker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_parallel_pipeline():
    # Create dummy PageModel objects
    dummy_pages = [
        PageModel(page_number=1, dimensions=[612, 792], elements=[]),
        PageModel(page_number=2, dimensions=[612, 792], elements=[]),
        PageModel(page_number=3, dimensions=[612, 792], elements=[]),
    ]
    config = Config()
    processed_pages = []
    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_page_worker, page.model_dump(), config): page.page_number
            for page in dummy_pages
        }
        logging.info(f"Submitted {len(futures)} pages for processing.")
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                result: ProcessResult = future.result()
                if result.error:
                    logging.error(f"Worker failed on page {page_num}: {result.error}")
                elif result.data:
                    logging.info(f"Successfully processed page {page_num}.")
                    processed_pages.append(result.data)
                else:
                    logging.warning(f"Worker for page {page_num} returned no data and no error.")
            except Exception as e:
                logging.critical(f"A critical error occurred fetching result for page {page_num}: {e}")
    processed_pages.sort(key=lambda p: p.page_number)
    logging.info(f"Test finished. Successfully processed {len(processed_pages)} pages.")
    for page in processed_pages:
        print(f"Page {page.page_number} processed: dimensions={page.dimensions}, elements={len(page.elements)}")

if __name__ == "__main__":
    test_parallel_pipeline() 