drvSetLock("SINGLE");
                                                                        SettingsDefinitions.ShootPhotoMode photoMode = SettingsDefinitions.ShootPhotoMode.BURST;
                                                                        //photoMode.PhotoBurstCount.BURST_COUNT_3;
                                                                        camera.setShootPhotoMode(photoMode, new djiCC().cb);

                                                                        //SettingsDefinitions.PhotoBurstCount burstCount = SettingsDefinitions.PhotoBurstCount.BURST_COUNT_3;
                                                                        //System.out.println("Count="+burstCount);
                                                                        //camera.setPhotoBurstCount(burstCount, new djiCC().cb);
                                                                        drvSpin();
                                                                        System.out.println("CaptureImageV2: shoot photo mode "+timeStamp(cmdStartTime));
