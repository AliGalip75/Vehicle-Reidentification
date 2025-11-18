import React, { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import { motion, AnimatePresence } from "framer-motion";
import { Meteors } from "@/components/ui/meteors"

import { Upload } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner"

export default function ReidSearch() {
  const [query, setQuery] = useState(null);
  const [queryPreview, setQueryPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [topK, setTopK] = useState(20);
  const [predictedId, setPredictedId] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [model, setModel] = useState("resnet");


  // -----------------------------
  // Fetch model performance
  // -----------------------------
  useEffect(() => {
    fetch(`http://localhost:8000/api/metrics/?model=${model}`)
      .then((res) => res.json())
      .then(setMetrics);
  }, [model]);


  const cardVariants = {
    hidden: { opacity: 0, y: 20, scale: 0.95 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        delay: i * 0.1,   // ← sırayla animasyon
        duration: 0.40,
        ease: "easeOut",
      },
    }),
  };

  // -----------------------------
  // Query gönder
  // -----------------------------
  const sendQuery = async () => {
    if (!query) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("image", query);
    formData.append("top_k", topK); // backend doğru format
    formData.append("model", model);


    const res = await fetch("http://localhost:8000/api/search/", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    setResults(data.results || []);

    // backend predicted_id döndürmüyor → top-1 result alınır
    const top1 = data.results?.[0];
    setPredictedId(top1 ? top1.id : null);

    setLoading(false);
  };

  return (
    <div className="p-8 flex gap-4 w-full">
      {/* SOL PANEL */}
      <div className="w-[500px] shrink-0 space-y-3 sticky top-8 self-start h-fit">
        <h2 className="text-2xl font-bold">Vehicle Re-ID</h2>

        {/* File input */}
        <label
          className="
            flex items-center justify-between 
            border border-gray-300 rounded-lg 
            px-4 py-3 cursor-pointer 
            hover:border-gray-500 transition
            bg-white
          "
        >
          <span className="text-gray-600 font-medium">
            {query ? query.name : "Select an image..."}
          </span>
          <Upload className="w-5 h-5 text-gray-600" />

          <input
            type="file"
            className="hidden"
            onChange={(e) => {
              if (!e.target.files?.[0]) return;
              const file = e.target.files[0];
              setQuery(file);
              setQueryPreview(URL.createObjectURL(file));
            }}
          />
        </label>

        {/* Top-K selector */}
        <Select onValueChange={(v) => setTopK(Number(v))} defaultValue="20">
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Result count" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="5">Top 5</SelectItem>
            <SelectItem value="10">Top 10</SelectItem>
            <SelectItem value="20">Top 20</SelectItem>
            <SelectItem value="40">Top 40</SelectItem>
            <SelectItem value="100">Top 100</SelectItem>
          </SelectContent>
        </Select>

        {/* Query Preview */}
        {queryPreview && (
          <Card>
            <CardContent className="p-3 space-y-3">
              <img
                src={queryPreview}
                alt="Query"
                className="rounded-lg border w-full h-40 object-cover"
              />

              {predictedId && (
                <div className="text-center text-lg font-semibold">
                  Query ID: <span>{predictedId}</span>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        <Select onValueChange={(v) => setModel(v)} defaultValue="resnet">
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select model" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="resnet">ResNet50-IBN</SelectItem>
            <SelectItem value="swin">Swin Transformer</SelectItem>
          </SelectContent>
        </Select>


        <Button onClick={sendQuery} className="w-full">
          {!loading ? "search" : <Spinner />}
        </Button>    

        {/* Metrics */}
        {metrics && (
          <Card>
            <CardContent className="p-4 space-y-2">
              <table class="table-fixed w-full text-left border-separate">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Rank-1</td>
                    <td>{metrics.rank1}</td>
                  </tr>
                  <tr>
                    <td>Rank-5</td>
                    <td>{metrics.rank5}</td>
                  </tr>
                  <tr>
                    <td>Rank-10</td>
                    <td>{metrics.rank10}</td>
                  </tr>
                  <tr>
                    <td>mAP</td>
                    <td>{metrics.mAP}</td>
                  </tr>
                  
                </tbody>
              </table>
            </CardContent>
          </Card>
        )}
      </div>

      {/* SAĞ PANEL */}
      <div className="flex-1">
        <h3 className="text-xl font-semibold mb-3">Top Results</h3>

        {loading ? (
          <div className="grid grid-cols-3 gap-4">
            {[...Array(9)].map((_, i) => (
              <Skeleton key={i} className="h-40 w-full rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-4">
            <AnimatePresence>
              {results.map((item, i) => (
                <motion.div
                  key={i}
                  custom={i}
                  initial="hidden"
                  animate="visible"
                  exit="hidden"
                  variants={cardVariants}
                >
                  <Card
                    className="cursor-pointer hover:shadow-xl transition"
                    onClick={() => setSelected(item)}
                  >
                    <CardContent className="p-3 space-y-2">
                      <img
                        src={`http://localhost:8000/${item.path}`}
                        className="rounded-lg border w-full h-40 object-cover"
                      />
                      <div className="text-sm font-medium flex justify-center">
                        <h3 className="font-sans text-xl">ID: {item.id}</h3>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* RESULT MODAL */}
      <Dialog open={!!selected} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Vehicle Details</DialogTitle>
          </DialogHeader>

          {selected && (
            <div className="space-y-3">
              <img
                src={`http://localhost:8000/${selected.path}`}
                alt=""
                className="rounded-lg border w-full object-cover"
              />

              <div className="text-sm font-medium flex justify-center">
                <h3 className="font-sans text-xl">ID: {selected.id}</h3>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
