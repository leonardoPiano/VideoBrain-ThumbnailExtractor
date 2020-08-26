# VideoBrain-ThumbnailExtractor
-DOD: yolo_thumbnail_processor.py
-FSI: scene_thumbnail_processor.py

***PyScene OSError: Video file(s) not found FIX***: 

PyScene controlla se il video esiste nella directory se non esiste lancia un errore. Per evitare cio bisogna eliminare le righe in cui
avviene il controllo  rispettivamente le linee 232 e 233 del file video_manager.py




